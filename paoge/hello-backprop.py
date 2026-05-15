import matplotlib.pyplot as plt

# 训练模型f(x) = w * x,  w真实值为5
def demo_backprop():
    # 1. 准备数据
    x = 2.0  # 输入
    y_target = 10.0  # 目标值

    # 2. 初始化权重 (随便猜一个数)
    w = 3.0
    learning_rate = 0.1  # 学习率 (步长)

    print(f"--- 开始训练 ---\n目标: 让 w * {x} 接近 {y_target}")
    print(f"初始权重 w = {w}\n")

    # 模拟 5 次迭代
    for epoch in range(1, 6):
        print(f"【第 {epoch} 轮】")

        # ==========================================
        # 步骤 1: 前向传播 (Forward Pass)
        # 算出当前的预测值和误差
        # ==========================================
        y_pred = w * x

        # 损失函数 (Loss): 使用均方误差 (y_pred - y_target)^2
        loss = (y_pred - y_target) ** 2

        print(f"  1. 前向计算: 预测值 = {y_pred:.4f}, Loss = {loss:.4f}")

        # ==========================================
        # 步骤 2: 反向传播 (Backward Pass) - 核心部分！
        # 我们需要计算 dLoss/dw (Loss 对 w 的导数)
        # 根据链式法则: dLoss/dw = (dLoss/dy_pred) * (dy_pred/dw)
        # ==========================================

        # A. Loss 对 预测值 的导数
        # Loss = (y_pred - y_target)^2  -> 导数是 2 * (y_pred - y_target)
        grad_pred = 2 * (y_pred - y_target)

        # B. 预测值 对 权重 的导数
        # y_pred = w * x  -> 对 w 求导就是 x
        grad_w_local = x

        # C. 链式法则：最终梯度
        w_grad = grad_pred * grad_w_local

        print(f"  2. 反向传播: 梯度计算过程 -> 2*({y_pred:.1f}-{y_target}) * {x} = {w_grad:.4f}")

        # ==========================================
        # 步骤 3: 梯度更新 (Update)
        # 新权重 = 旧权重 - (学习率 * 梯度)
        # ==========================================
        w = w - (learning_rate * w_grad)
        print(f"  3. 参数更新: w = {w + learning_rate * w_grad:.4f} - ({learning_rate} * {w_grad:.4f}) = {w:.4f}")
        print("-" * 30)

    print(f"\n最终结果: w = {w:.4f} (理想值应该是 5.0)")


if __name__ == "__main__":
    demo_backprop()