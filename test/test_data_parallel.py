import torch
from collections import namedtuple

def process(input) -> int:
    return (input + 1, input + 2)


if __name__ == '__main__':
    
    LossOutput = namedtuple('LossOutput', ['loss', 'accuracy'])
    loss_output = LossOutput(loss=1, accuracy=0)
    res = map(process, loss_output)
    res = zip(*res)
    for item in res:
        print(item)