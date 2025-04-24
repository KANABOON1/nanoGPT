import torch

if __name__ == '__main__':
    parameters = [torch.tensor([1, 2, 3, 4, 5]), torch.tensor([6, 7, 8])]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    for group in optimizer.param_groups:   # 参数组
        print(group)