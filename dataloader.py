import tiktoken
import torch

class DataLoaderLite:
   def __init__(self, B: int, T: int):
      self.B = B
      self.T = T

      with open('data/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
         data = f.read()
      enc = tiktoken.get_encoding("gpt2")
      tokens = enc.encode(data)
      self.tokens = torch.tensor(tokens)
      print(f"Load {len(self.tokens)} tokens.")
      print(f"1 epoch = {len(self.tokens) // (B * T)} batches.")
      
      self.current_position = 0
   def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
      # get the current batch
      B, T = self.B, self.T
      buf = self.tokens[self.current_position: self.current_position + B * T + 1]

      # NOTE - 构造错位生成 batch X 以及 batch y
      x = buf[:-1].view(B, T)
      y = buf[1:].view(B, T) 

      # forward the current position
      self.current_position += B * T
      if self.current_position + B * T >= len(self.tokens):
         self.current_position = 0
      
      return x, y

