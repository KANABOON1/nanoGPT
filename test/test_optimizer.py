import torch
import torch.nn.functional as F

if __name__ == '__main__':
    predicted = torch.tensor([[0.9, 0.7, 0.2]])
    target = torch.tensor([[0.1, 0.1, 0.8]])

    loss = F.cross_entropy(predicted, target)
    print(loss)
