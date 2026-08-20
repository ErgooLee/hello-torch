import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# 解决 Mac 系统中文字体缺失及负号显示问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False


def train_sin_x():
    # 1. 目标函数：复杂的连续非线性函数 y = sin(x)
    x = torch.linspace(-np.pi, np.pi, 200).unsqueeze(1)
    y = torch.sin(x)

    # 2. 仅有 1 个隐藏层，但塞入 500 个神经元（参数足够多）
    model = nn.Sequential(
        nn.Linear(1, 500),  # 隐藏层：500 个节点
        nn.ReLU(),  # 激活函数
        nn.Linear(500, 1),  # 输出层加权汇总
    )

    # 3. 快速训练
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for _ in range(1000):
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"训练完成！最终拟合误差 MSE Loss: {loss.item():.6f}")


def relu(x):
    return np.maximum(0, x)


def show_uat():
    x = np.linspace(-1, 5, 500)

    # 定义 3 个隐藏层神经元 (产生 3 个在不同位置拐弯的折线)
    h1 = relu(x - 0)  # 在 x=0 处起折
    h2 = relu(x - 2)  # 在 x=2 处起折
    h3 = relu(x - 4)  # 在 x=4 处起折

    # 输出层做加权求和: y = 1*h1 - 2*h2 + 1*h3 (造出一个三角形小山峰)
    y_hat = 1.0 * h1 - 2.0 * h2 + 1.0 * h3

    # 绘图展示
    plt.figure(figsize=(10, 5))
    plt.plot(x, h1, '--', label='Neuron 1: ReLU(x - 0)', alpha=0.6)
    plt.plot(x, -2 * h2, '--', label='Neuron 2: -2 * ReLU(x - 2)', alpha=0.6)
    plt.plot(x, h3, '--', label='Neuron 3: ReLU(x - 4)', alpha=0.6)
    plt.plot(x, y_hat, 'r-', linewidth=3, label='Combined Output: y = h1 - 2*h2 + h3')

    plt.axhline(0, color='black', linewidth=0.8)
    plt.title('How 3 ReLUs form a localized "Hat" Function', fontsize=12)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_sin_x()
    show_uat()
