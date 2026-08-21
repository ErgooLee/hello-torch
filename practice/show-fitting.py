import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

# 解决 Mac 系统中文字体缺失及负号显示问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False


def test_fitting():
    # 1. 准备数据集：划分训练集和测试集
    torch.manual_seed(42)
    np.random.seed(42)

    n_train = 15
    x_train = torch.rand(n_train, 1) * 2 * np.pi - np.pi
    y_train = torch.sin(x_train) + torch.randn(x_train.size()) * 0.25

    x_test = torch.linspace(-np.pi, np.pi, 200).unsqueeze(1)
    y_test = torch.sin(x_test)

    # 2. 定义不同模型容量与训练配置，分别演示：欠拟合、合适拟合、过拟合
    configs = [
        {
            "name": "欠拟合 (Underfitting)",
            "model": nn.Sequential(nn.Linear(1, 1)),
            "epochs": 500,
            "lr": 0.01
        },
        {
            "name": "合适拟合 (Good Fit)",
            "model": nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1)),
            "epochs": 2000,
            "lr": 0.01
        },
        {
            "name": "过拟合 (Overfitting)",
            "model": nn.Sequential(
                nn.Linear(1, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ),
            "epochs": 8000,
            "lr": 0.005
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, cfg in enumerate(configs):
        model = cfg["model"]
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        criterion = nn.MSELoss()

        for _ in range(cfg["epochs"]):
            pred_train = model(x_train)
            loss_train = criterion(pred_train, y_train)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_train_final = model(x_train)
            pred_test_final = model(x_test)
            loss_train_final = criterion(pred_train_final, y_train).item()
            loss_test_final = criterion(pred_test_final, y_test).item()

        print(f"[{cfg['name']}]")
        print(f"  训练集 MSE Loss: {loss_train_final:.6f}")
        print(f"  测试集 MSE Loss: {loss_test_final:.6f}")

        # 绘图对比
        ax = axes[idx]
        ax.scatter(x_train.numpy(), y_train.numpy(), color='red', label='训练集 (含噪)', zorder=5)
        ax.plot(x_test.numpy(), y_test.numpy(), 'g--', label='真实函数 sin(x)', alpha=0.7)
        ax.plot(x_test.numpy(), pred_test_final.numpy(), 'b-', linewidth=2, label='模型预测')
        ax.set_title(f"{cfg['name']}\nTrain Loss: {loss_train_final:.4f} | Test Loss: {loss_test_final:.4f}",
                     fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()


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
    test_fitting()
    # show_uat()
