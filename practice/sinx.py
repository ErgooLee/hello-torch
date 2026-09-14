import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def show_sin_x():
    # 1. 设置随机种子，保证实验可复现
    torch.manual_seed(42)
    np.random.seed(42)

    # 2. 生成训练与测试数据：[-pi, pi]
    x_train = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1).astype(np.float32)
    y_train = np.sin(x_train)  # 目标函数：sin(x)，也可以替换为 cos(x) 等

    # 转化为 PyTorch Tensor
    X = torch.from_numpy(x_train)
    Y = torch.from_numpy(y_train)

    # 3. 定义多层感知机 (MLP) 架构
    class TrigMLP(nn.Module):
        def __init__(self, hidden_dim=64):
            super(TrigMLP, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Tanh(),  # 光滑激活函数，特别适合回归周期平滑曲线
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)  # 输出层为线性输出
            )

        def forward(self, x):
            return self.net(x)

    model = TrigMLP(hidden_dim=64)

    # 4. 定义损失函数与优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 5. 模型训练循环
    epochs = 800
    for epoch in range(1, epochs + 1):
        # 前向传播
        y_pred = model(X)
        loss = criterion(y_pred, Y)

        # 反向传播与梯度更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch [{epoch:4d}/{epochs}] - Loss (MSE): {loss.item():.6f}")

    # 6. 推理与可视化验证
    model.eval()
    with torch.no_grad():
        x_test = np.linspace(-np.pi, np.pi, 500).reshape(-1, 1).astype(np.float32)
        y_test_true = np.sin(x_test)
        y_test_pred = model(torch.from_numpy(x_test)).numpy()

    # 绘制拟合对比图
    plt.figure(figsize=(9, 5))
    plt.plot(x_test, y_test_true, 'k--', label='True sin(x)', linewidth=2)
    plt.plot(x_test, y_test_pred, 'r-', label='MLP Prediction', linewidth=2)
    plt.scatter(x_train, y_train, color='blue', s=10, alpha=0.5, label='Train Points')
    plt.title("MLP Fitting sin(x) in [-π, π]")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    show_sin_x()