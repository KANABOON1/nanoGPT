import time
import math
import torch
from torch.nn import functional as F

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

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:2'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():   # mac
        device = 'mps'

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    
    # NOTE - batch size 与所有训练有关的超参数高度相关
    batch_size = 524288 # ~5M, beautiful number: 2^19
    B, T = 16, 1024
    assert batch_size % (B * T) == 0
    grad_accum_steps: int = batch_size // (B * T)
    train_loader = DataLoaderLite(B=B, T=T)
    
    # NOTE - A100-ampere: 使用 TF32 加速
    torch.set_float32_matmul_precision("high")  
    
    # NOTE - really beautiful number!
    # NOTE - 相比于原先的 GPT2(vacab_size=50257), vacab_size=50304 仅仅增加了一点 size(token embedding tensor)
    #        在优化时, 增加的部分 embedding 的概率将被优化接近 0 (例如Shakespeare数据集仅涉及到10000个单词, 剩下的40000个单词的输出概率接近于0)
    model: GPT = GPT(GPTConfig(vocab_size=50304))    
    model.eval()
    model = model.to(device)
    model = torch.compile(model)   # 静态编译model, 优化模型推理速度: 减少了 HBM 与 GPU 之间的差距
    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)
    
    use_amp = True
    for step in range(max_steps):
        t0 = time.time()

        optimizer.zero_grad()
        loss_accum = 0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            # NOTE - In this context, use automatic mixed precision: transfer some tensor to bf16, which is faster than fp32
            # NOTE - 不改变模型本身的参数, 而是改变用于计算时的参数, 并且允许输出时低精度的
            # NOTE - 使用 bf16 而不使用 fp16, 因为 bf16 能够表示的范围比 fp16 更广, 不容易产生下溢问题(下溢:当小于小数所能表示的最小数字时, 会变为0)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp): 
                logits, loss = model(x, y)
            
            # let the gradient backward and accumulated per micro_step
            # NOTE - scaling down the loss, 仅仅使用一个 object 的 loss
            loss = loss / grad_accum_steps  
            loss_accum += loss.detach()
            loss.backward()

        # NOTE - 裁剪梯度总范数, 使得总范数 <= max_norm, 防止梯度爆炸
        # NOTE - 如果范数 cur_norm 超过 max_norm, 则对所有梯度进行缩放: 都乘以 max_norm / cur_norm
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
        
        # backward 使用全精度
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 
        optimizer.step()

        torch.cuda.synchronize()    # finish all the works that's scheduled to run
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
        tokens_per_second = tokens_processed / dt

        print(f"step {step} | loss = {loss_accum.item()} | lr = {lr:.4e} | dt = {1000 * dt:.2f}ms | norm = {norm: .4f} | tokens_per_second = {tokens_per_second:.2f}")