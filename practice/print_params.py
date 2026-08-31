import torch
from torch import nn


def print_params():
    net = nn.Sequential(nn.Linear(10, 20),
                        nn.ReLU(),
                        nn.Linear(20, 1))
    x = torch.randn(2, 10)
    net(x)
    print(net[2].state_dict())
    print(type(net[2].bias))
    print(net[2].bias)
    print(net[2].bias.data)
    print(net[2].bias.grad)

if __name__ == '__main__':
    print_params()