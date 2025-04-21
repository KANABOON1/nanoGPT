from dataclasses import dataclass
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
    def __init__(self, config):
        super().__init__()




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