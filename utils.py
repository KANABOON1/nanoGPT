import numpy as np
import torch
def load_tokens(filename: str) -> torch.Tensor:

    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.int32)
    return ptt