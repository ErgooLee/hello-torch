import torch

# ================= 1. 准备数据 =================
# 我们设定真实的 w 和 b
true_w = torch.tensor([[2.0], [-3.4]])  # 形状为 (2, 1) 的列向量
true_b = 4.2

# 随机生成 1000 个特征样本 X，形状为 (1000, 2)
X = torch.normal(0, 1, (1000, 2))

# 根据公式 y = Xw + b 算出真实标签，并加上一点高斯噪声
y = torch.matmul(X, true_w) + true_b
y += torch.normal(0, 0.01, y.shape)


# ================= 2. 初始化要学习的参数 =================
# 随机初始化我们的预测权重 w 和偏差 b，并开启梯度追踪
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# ================= 3. 训练循环 =================
lr = 0.1  # 学习率
epochs = 50  # 迭代次数

for epoch in range(epochs):
    # 步骤 A: 前向传播（用当前的 w 和 b 预测所有的 y）
    y_hat = torch.matmul(X, w) + b

    # 步骤 B: 计算所有样本的平均损失 (均方误差)
    loss = ((y_hat - y) ** 2 / 2).mean()

    # 步骤 C: 反向传播（计算损失关于 w 和 b 的梯度）
    loss.backward()

    # 步骤 D: 更新参数（梯度下降）并清空梯度
    with torch.no_grad():
        w -= lr * w.grad  # w 沿着梯度反方向移动
        b -= lr * b.grad  # b 沿着梯度反方向移动

        w.grad.zero_()  # 必须清空梯度，否则下次计算会累加
        b.grad.zero_()

    # 每隔 10 次迭代打印一次当前的损失值
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1:2d}, Loss: {loss.item():.6f}")

# ================= 4. 结果对比 =================
print("\n--- 训练结束 ---")
print(f"真实 w:\n{true_w.tolist()}")
print(f"学到的 w:\n{w.tolist()}")
print(f"真实 b: {true_b}")
print(f"学到的 b: {b.item():.4f}")