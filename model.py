from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 256    # context size
    vocab_size: int = 65     # vocabulary size
    n_layer: int = 6         # number of layers
    n_head: int = 6          # multi-head attention
    n_embd: int = 384        # embedding dimension
    dropout: float = 0.0


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    """masked multihead attention"""
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.flash = hasattr(nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size).
                                                    view(1, 1, config.block_size, config.block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimension
        
        # 构造多头注意力矩阵
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # [B, nh, T, nd]
        k = k.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)

        if self.flash:
            y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:   # manually implement multi-head attention
            # multihead attention score
            att: torch.Tensor = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            # process attention scores
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)  # note: 必须先softmax再做dropout, 否则dropout会打乱原有的权重分布
            att = self.attn_dropout(att)

            # values
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y



            


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)   # 层归一化
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE - 相比于 `Attention is all you need`原论文中的block架构, 这里将residual产生的add与layer norm分离
        # 这样做的好处是: gradient从top -> bottom, 不会收到中间的layer norm的影响(梯度爆炸?)
        # 此时residual完全从模块中独立
        x = x + self.attn(self.ln1(x))   # aggregation function (information collected between the tokens): communicate
        x = x + self.mlp(self.ln2(x))    # mapping function: think individually

        return x

class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),                # output embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),                # position embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),   # layers(blocks)
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)