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
        
    train_loader = DataLoaderLite(B=16, T=1024)
    
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
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        
        # NOTE - In this context, use automatic mixed precision: transfer some tensor to bf16, which is faster than fp32
        # NOTE - 使用 bf16 而不使用 fp16, 因为 bf16 能够表示的范围比 fp16 更广, 不容易产生下溢问题
        # NOTE - 下溢: 当小于小数所能表示的最小数字时, 会变为0
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp): 
            logits, loss = model(x, y)
        
        # backward 使用全精度
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 裁剪总范数, 使得总范数 <= max_norm, 防止梯度爆炸
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()    # finish all the works that's scheduled to run
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_second = train_loader.B * train_loader.T / (t1 - t0)

        print(f"step {step} | loss = {loss.item()} | lr = {lr:.4e} | dt = {dt:.2f}ms | norm = {norm: .4f} | tokens_per_second = {tokens_per_second:.2f}")

    # logits, loss = model(x, y)
    
    # print(logits.shape)
    # print(loss)   # NOTE - loss = 10.8893, 大约为 - log(1 / vocab_size), 说明参数分布均匀, 模型初始化时输出的每一个vocab的概率几乎相等
    # print(loss.shape)

    # while x.size(1) < max_length:
    #     with torch.no_grad():
    #         logits = model(x)[:, -1, :]    # [B, vocab_size]
    #         probs = F.softmax(logits, dim=-1)
            
    #         # topk sampling of 50 (huggingface pipeline 默认设置)
    #         topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)   # [B, 50]
            
    #         # 从 samples 中按照概率随机抽取一个token
    #         ix = torch.multinomial(topk_probs, num_samples=1)    # [B, 1]
        
    #         # gather the corresponding indices
    #         xcol = torch.gather(topk_indices, dim=-1, index=ix)

    #         # append to the sequence
    #         x = torch.cat([x, xcol], dim=1)
    
    # for i in range(num_return_sequences):
    #     tokens = x[i][:max_length].tolist()
    #     decoded = enc.decode(tokens)
    #     print("> ", decoded)