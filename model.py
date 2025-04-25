import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import inspect
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

@dataclass
class GPTConfig:
    block_size: int = 1024    # context size
    vocab_size: int = 50257     # vocabulary size: 50000 BPE merges + 256 bytes tokens + 1 <|endoftxt|>
    n_layer: int = 12         # number of layers
    n_head: int = 12          # multi-head attention
    n_embd: int = 768        # embedding dimension
    dropout: float = 0.0
    bias: bool = False


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # NOTE - 在 project layer 对 resiual layers 产生的权重进行缩放
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    """masked multihead attention"""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.bias = config.bias

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

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
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))  # 变为负无穷, 经过 softmax 之后 attention score 变为0
            att = F.softmax(att, dim=-1)  # note: 必须先softmax再做dropout, 否则dropout会打乱原有的权重分布

            # TODO - attention 应该有 dropout
            # att = F.dropout(att, p=self.dropout, training=self.training)

            # values
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.c_proj(y)

        return y            


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)   # 层归一化
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE - 相比于 `Attention is all you need`原论文中的block架构, 这里将residual产生的add与layer norm分离
        # 这样做的好处是: gradient从top -> bottom, 不会收到中间的layer norm的影响(梯度爆炸?)
        # 此时residual完全从模块中独立
        x = x + self.attn(self.ln_1(x))   # aggregation function (information collected between the tokens): communicate
        x = x + self.mlp(self.ln_2(x))    # mapping function: think individually

        return x

class GPT(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),                # output embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),                # position embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),   # layers(blocks)
            ln_f = nn.LayerNorm(config.n_embd),                                  # layer_norm final
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # NOTE - 针对残差连接的数量进行标准化
                # NOTE - 每一个 layer 对应 2 个残差连接
                std *= (2 * self.config.n_layer)**-0.5 
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx: torch.Tensor, target: torch.Tensor=None) -> torch.Tensor:
        """GPT 根据输入的tokens预测下一个token

        Args:
            idx (torch.Tensor): 输入的 tokens, dim = [B, T]

        Returns:
            torch.Tensor: logits tensor, dim = [B, T, vocab_size], 每一个在输入sequence中的 token 都会计算出下一个token的概率
        """
        # process inputs
        B, T = idx.size()   # batch size, sequence length
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # forward the token and position embeddings
        # NOTE - 按照输入的 idx 所在设备处理, 如果 idx 设备与模型设备不一致, 则后续会报错
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = pos_emb + tok_emb

        # forward the blocks of transformers
        for block in self.transformer.h:
            x = block(x)
        # forward the final layer norm and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # calculate loss
        loss = None
        if target is not None:
            # ignore_index=-1, 忽略掉pad的loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1)  
        return logits, loss
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True

        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        
        # empty model 
        model = GPT(GPTConfig(**config_args))
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        
        sd_hf = GPT2LMHeadModel.from_pretrained(model_type).state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]

        transpose = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys) == len(sd_keys_hf), f"mismatched keys: {len(sd_keys)} != {len(sd_keys_hf)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transpose):
                # NOTE - transpose conv1D to linear
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that requires grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D dimensional will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},  # NOTE - 对二维矩阵进行 weight decay, 防止 over-fitting
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        # print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device != 'cpu' and device != 'mps'
        # print(f"using fused AdamW: {use_fused}")
        optimizer = AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)   # fused implementation

        return optimizer



