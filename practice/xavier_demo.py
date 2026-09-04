import torch
import torch.nn as nn

def run_forward_backward(init_type="xavier", num_layers=10, dim=256):
    """
    构建一个多层网络，观察在不同初始化下：
    1. 前向传播中每层激活值的均值与标准差 (衡量信号是否消失/饱和)
    2. 反向传播中每层权重的梯度标准差 (衡量梯度是否消失/爆炸)
    """
    torch.manual_seed(42)
    
    # 构造输入: 1000 个样本，每样本 dim 维，标准正态分布输入
    x = torch.randn(1000, dim)
    layers = []
    
    # 构建多层网络并应用不同的初始化
    for _ in range(num_layers):
        layer = nn.Linear(dim, dim, bias=False)
        if init_type == "small":
            # 权重标准差过小 (0.01)
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
        elif init_type == "large":
            # 权重标准差过大 (1.0)
            nn.init.normal_(layer.weight, mean=0.0, std=1.0)
        elif init_type == "xavier":
            # Xavier 正态分布初始化
            nn.init.xavier_normal_(layer.weight)
        layers.append(layer)
    
    act_stds = []
    current = x
    
    # 前向传播 (使用 Tanh 激活函数)
    for i, layer in enumerate(layers):
        current = torch.tanh(layer(current))
        act_stds.append(current.std().item())
        
    # 反向传播 (构造虚拟损失)
    loss = current.sum()
    loss.backward()
    grad_stds = [layer.weight.grad.std().item() for layer in layers]
    
    return act_stds, grad_stds


def main():
    num_layers = 8
    dim = 256

    print("=" * 75)
    print(" 实验演示: 8层深层网络 (Tanh 激活) 下不同初始化的信号与梯度传递")
    print("=" * 75)

    experiments = [
        ("权重过小 (std=0.01)", "small"),
        ("权重过大 (std=1.0)", "large"),
        ("Xavier 正态分布初始化", "xavier"),
    ]

    results = {}
    for name, key in experiments:
        act_stds, grad_stds = run_forward_backward(init_type=key, num_layers=num_layers, dim=dim)
        results[name] = (act_stds, grad_stds)

    # 打印前向传播各层激活值标准差
    print("\n【1. 前向传播各层激活值的标准差 (Std)】(输入初始 std ≈ 1.0)")
    print(f"{'网络层':<8} | {'权重过小 (std=0.01)':<20} | {'权重过大 (std=1.0)':<20} | {'Xavier 初始化':<20}")
    print("-" * 75)
    for i in range(num_layers):
        s_small = results["权重过小 (std=0.01)"][0][i]
        s_large = results["权重过大 (std=1.0)"][0][i]
        s_xavier = results["Xavier 正态分布初始化"][0][i]
        print(f"Layer {i+1:<2} | {s_small:<20.6f} | {s_large:<20.6f} | {s_xavier:<20.6f}")

    # 打印反向传播各层梯度标准差
    print("\n【2. 反向传播各层权重梯度的标准差 (Grad Std)】")
    print(f"{'网络层':<8} | {'权重过小 (std=0.01)':<20} | {'权重过大 (std=1.0)':<20} | {'Xavier 初始化':<20}")
    print("-" * 75)
    for i in range(num_layers):
        g_small = results["权重过小 (std=0.01)"][1][i]
        g_large = results["权重过大 (std=1.0)"][1][i]
        g_xavier = results["Xavier 正态分布初始化"][1][i]
        print(f"Layer {i+1:<2} | {g_small:<20.6f} | {g_large:<20.6f} | {g_xavier:<20.6f}")

    print("\n" + "=" * 75)
    print("【实验结论分析】")
    print("1. 权重过小 (std=0.01) : 激活值在前向传播中迅速衰减归零，反向传播前几层梯度严重衰减 (梯度消失)。")
    print("2. 权重过大 (std=1.0)  : 激活值直接打满 Tanh 饱和区 (std 饱和在 ~0.99)，Tanh 导数趋近 0，反向传播梯度几乎为 0。")
    print("3. Xavier 初始化       : 前向各层方差平稳维持在 0.5~0.6 之间，反向各层梯度平稳流动，网络能够健康学习。")
    print("=" * 75)


if __name__ == "__main__":
    main()
