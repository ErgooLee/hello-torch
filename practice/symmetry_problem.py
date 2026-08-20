import torch
import torch.nn as nn


def show_symmetry_problem1():

    hidden_layer = nn.Linear(4, 3)
    output_layer = nn.Linear(3, 1)
    net = nn.Sequential(hidden_layer, nn.ReLU(), output_layer)

    # 2. 全部设为相同常数 0.5
    nn.init.constant_(hidden_layer.weight, 0)
    nn.init.constant_(hidden_layer.bias, 0)
    nn.init.constant_(output_layer.weight, 0)
    nn.init.constant_(output_layer.bias, 0)

    x = torch.randn(2, 4)
    y = torch.tensor([[1.0], [2.0]])

    loss = nn.MSELoss()(net(x), y)
    loss.backward()

    print("=== 场景 2：全常数 (0.5) 初始化时的隐藏层梯度 ===")
    print(hidden_layer.weight.grad)



def show_symmetry_problem2():

    hidden_layer = nn.Linear(4, 3)
    output_layer = nn.Linear(3, 1)
    net = nn.Sequential(hidden_layer, nn.ReLU(), output_layer)

    # 2. 全部设为相同常数 0.5
    nn.init.constant_(hidden_layer.weight, 0.5)
    nn.init.constant_(output_layer.weight, 0.5)
    nn.init.constant_(hidden_layer.bias, 0.0)
    nn.init.constant_(output_layer.bias, 0.0)

    x = torch.randn(2, 4)
    y = torch.tensor([[1.0], [2.0]])

    loss = nn.MSELoss()(net(x), y)
    loss.backward()

    print("=== 场景 2：全常数 (0.5) 初始化时的隐藏层梯度 ===")
    print(hidden_layer.weight.grad)

if __name__ == "__main__":
    show_symmetry_problem1()
    show_symmetry_problem2()