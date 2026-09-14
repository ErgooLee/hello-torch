import torch
import torch.nn as nn


def run_experiment(features, init_type):
    """
    init_type 可选:
      - 'large' : 初始方差过大 (std=1.0) -> 导致激活饱和、梯度消失
      - 'small' : 初始方差过小 (std=0.01) -> 导致前向信号坍缩至 0
      - 'xavier': 使用 Xavier/Glorot Normal 初始化 -> 保持方差稳定
    """
    layers = []
    for _ in range(10):
        linear = nn.Linear(100, 100, bias=False)

        # 不同的初始化策略
        if init_type == 'large':
            nn.init.normal_(linear.weight, mean=0.0, std=1.0)
        elif init_type == 'small':
            nn.init.normal_(linear.weight, mean=0.0, std=0.1)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(linear.weight)

        layers.append(linear)



    print(f"\n{'=' * 20} 策略: {init_type.upper()} {'=' * 20}")

    # 前向传播并记录每一层输出的方差
    h = features
    activation_vars = []
    for i, layer in enumerate(layers):
        # 线性变换 + Tanh 激活函数
        h = torch.tanh(layer(h))
        # 保留中间变量的梯度以便观察反向传播
        h.retain_grad()
        activation_vars.append((i + 1, h))

    # 打印前向激活方差
    for layer_idx, out in activation_vars:
        if layer_idx in [1, 2, 5, 10]:  # 重点观察第 1, 2, 5, 10 层
            print(f"第 {layer_idx:2d} 层激活输出: 方差 = {out.var():.6f}")

    # 反向传播：假设最终损失是最后一层输出的和
    loss = h.sum()
    loss.backward()

    # 打印第一层的反向梯度模长/方差
    first_layer_grad = layers[0].weight.grad
    print(f"第 1 层权重梯度: 均值 = {first_layer_grad.mean():.6e}, 方差 = {first_layer_grad.var():.6e}")


if __name__ == '__main__':
    # 设置随机种子保证结果可复现
    torch.manual_seed(42)

    # 构造输入数据：均值 0，方差 1
    features = torch.randn(200, 100, requires_grad=True)
    print(f"输入层: 均值 = {features.mean():.4f}, 方差 = {features.var():.4f}")

    # 1. 方差过小
    run_experiment(features, 'small')
    # 2. 方差过大
    run_experiment(features, 'large')
    # 3. Xavier 初始化
    run_experiment(features, 'xavier')