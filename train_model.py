import os
import time
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig
from dataloader import DataLoaderLite

max_lr = 6e-4
min_lr = 0.1 * max_lr
warmup_steps = 10
max_steps = 50

# NOTE - 虽然 torch 有相关的实现, 但是与其使用 black box, 也可以使用自己实现的东西
def get_lr(step: int) -> float:
    """根基 step id (从 0 开始)计算当前的 lr
    """
    # 1) linear warmup for warmup_ites step
    if step < warmup_steps:
        return (step + 1) * max_lr / warmup_steps  # 防止 step = 0 时 lr = 0
    
    # 2) if step > max_steps, return min_lr
    elif step >= max_steps:
        return min_lr
    
    # 3) in between warmup_steps and max_steps, use cosine decay down to min learning rate
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
    

if __name__ == '__main__':

    # NOTE - set up DDP (distributed data parallel)
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), 'DDP need cuda'
        dist.init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(ddp_local_rank)
        master_process = ddp_rank == 0       # 用于区分是否是主进程
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True

        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda:2'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():   # mac
            device = 'mps'

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    
    # NOTE - batch size 与所有训练有关的超参数高度相关
    batch_size = 524288 # effective batch size, ~5M, beautiful number: 2^19
    B, T = 16, 1024
    assert batch_size % (B * T * ddp_world_size) == 0
    grad_accum_steps: int = batch_size // (B * T * ddp_world_size)  # 每一个进程只需要 origin_accum_step / wordsize 个 accum_steps
    if master_process:
        print(f"Micro steps in each batch: {grad_accum_steps}")
    
    # NOTE - 需要确保不同的进程得到的训练数据不同, 否则会造成无意义的消耗(同一个 batch 中有多个进程得到了相同的 batch)
    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)
    
    # NOTE - A100-ampere: 使用 TF32 加速
    torch.set_float32_matmul_precision("high")  
    
    # NOTE - really beautiful number!
    # 相比于原先的 GPT2(vacab_size=50257), vacab_size=50304 仅仅增加了一点 size(token embedding tensor)
    # 在优化时, 增加的部分 embedding 的概率将被优化接近 0 (例如Shakespeare数据集仅涉及到10000个单词, 剩下的40000个单词的输出概率接近于0)
    model: GPT = GPT(GPTConfig(vocab_size=50304))    
    model.eval()
    model = model.to(device)
    model = torch.compile(model)   # 静态编译model, 优化模型推理速度: 减少了 HBM 与 GPU 之间的差距
    if master_process:
        print('---------------Finish Compiling Model!----------------')

    if ddp:
        # NOTE - synchronize and average the gradients using Ring-AllReduce
        model: DDP = DDP(model, device_ids=[ddp_local_rank], output_device=ddp_local_rank)
        if master_process:
            print('---------------Use Distributed Data Parallel!----------------')
    raw_model = model.module if ddp else model
    
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)
    
    use_amp = True
    for step in range(max_steps):
        t0 = time.time()

        optimizer.zero_grad()
        loss_accum = 0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            # NOTE - In this context, use automatic mixed precision: transfer some tensor to bf16, which is faster than fp32
            # 不改变模型本身的参数, 而是改变用于计算时的参数, 并且允许输出时低精度的
            # 使用 bf16 而不使用 fp16, 因为 bf16 能够表示的范围比 fp16 更广, 不容易产生下溢问题(下溢:当小于小数所能表示的最小数字时, 会变为0)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp): 
                logits, loss = model(x, y)
            # let the gradient backward and accumulated per micro_step
            # NOTE - scaling down the loss, 仅仅使用一个 object 的 loss
            loss = loss / grad_accum_steps  
            loss_accum += loss.detach()

            # NOTE - 使用 require_backward_grad_sync 标签控制 loss.backward() 时是否需要使用 AllReduce 操作
            # 因为一次 AllReduce 操作开销很大, 在 accumulate_grads 操作中实际上只有最后一次 loss.backward() 需要使用 AllReduce 操作进行不同 process 之间梯度的同步.
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)  
            # NOTE - DDP 模式下 (require_backward_grad_sync == True), 每一个进程在 loss.backward() 时都会触发梯度的 AllReduce 操作
            # 该操作使用 Ring-AllReduce 对每一个进程的梯度进行同步, 确保每一个进程上的梯度相同
            # 最后再每一个进程单独执行 optimizer.step().
            loss.backward()   
        
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # NOTE - 裁剪梯度总范数, 使得总范数 <= max_norm, 防止梯度爆炸
        # 如果范数 cur_norm 超过 max_norm, 则对所有梯度进行缩放: 都乘以 max_norm / cur_norm
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
        
        # backward 使用全精度
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 
        optimizer.step()

        torch.cuda.synchronize()    # finish all the works that's scheduled to run
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_second = tokens_processed / dt
        
        if master_process:
            print(f"step {step} | loss = {loss_accum.item()} | lr = {lr:.4e} | dt = {1000 * dt:.2f}ms | norm = {norm: .4f} | tokens_per_second = {tokens_per_second:.2f}")
    
    if ddp:
        dist.destroy_process_group()