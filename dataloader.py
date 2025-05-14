import torch
from typing import Literal
import os

from utils import load_tokens

class DataLoaderLite:
   def __init__(self, B: int, T: int, process_rank: int, num_processes: int, split: Literal['train', 'val']):
      # process input
      assert split in ['train', 'val']
      
      self.B = B
      self.T = T
      self.process_rank = process_rank
      self.num_processes = num_processes

      data_root = 'edu_fineweb10B'
      shards = os.listdir(data_root)
      shards = [s for s in shards if split in s]
      shards = sorted(shards)
      shards = [os.path.join(data_root, s) for s in shards]
      self.shards = shards
      assert len(shards) > 0, 'No shards found for split {split}'
      
      if process_rank == 0:
         print(f"found {len(shards)} for split {split}")
      self.reset()
   
   def reset(self):
      self.current_shard = 0
      self.tokens = load_tokens(self.shards[self.current_shard])
      self.current_position = self.B * self.T * self.process_rank

   def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
      # get the current batch
      B, T = self.B, self.T
      buf = self.tokens[self.current_position: self.current_position + B * T + 1]

      # NOTE - 构造错位生成 batch X 以及 batch y
      x = buf[:-1].view(B, T)
      y = buf[1:].view(B, T) 

      # forward the current position
      self.current_position += B * T * self.num_processes
      if self.current_position + B * T * self.num_processes >= len(self.tokens):
         self.current_shard = (self.current_shard + 1) % len(self.shards)
         self.tokens = load_tokens(self.shards[self.current_shard])
         self.current_position = B * T * self.process_rank
      
      return x, y

