import os
import torch
from torch.nn import functional as F

from model import GPT, GPTConfig
from dataloader import DataLoaderLite

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # hyperparameters
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'

    num_return_sequences = 5
    max_length = 30
    
    train_loader = DataLoaderLite(B=4, T=32)
    
    # model
    model = GPT(GPTConfig())
    model.eval()
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(50):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()   

        print(f"step {i}, loss = {loss.item()}")

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