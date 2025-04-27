import torch
def test_torch_lt():
    x = torch.randn(5)
    res = torch.lt(x.sum(), 0)
    print(res)
    print(type(res))

if __name__ == '__main__':
    test_torch_lt()
