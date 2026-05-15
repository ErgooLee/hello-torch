import torch

def pytorch_demo():
    # ==========================================
    # 1. 准备数据
    # 在 PyTorch 中，数据必须是 Tensor (张量)
    # ==========================================
    x = torch.tensor(2.0)
    y_target = torch.tensor(10.0)

    # ==========================================
    # 2. 初始化权重
    # requires_grad=True 是核心：告诉 PyTorch 追踪这个变量的所有计算
    # 相当于告诉系统："这不仅仅是个数字 3.0，请记住它参与的所有公式"
    # ==========================================
    w = torch.tensor(3.0, requires_grad=True)
    learning_rate = 0.1

    print(f"--- PyTorch 训练开始 ---\n目标: 让 w * {x.item()} 接近 {y_target.item()}")
    print(f"初始权重 w = {w.item()}\n")

    for epoch in range(1, 6):
        print(f"【第 {epoch} 轮】")

        # ==========================================
        # 步骤 1: 前向传播 (Forward)
        # 写法和普通数学公式一样，但 PyTorch 在后台构建了“计算图”
        # ==========================================
        y_pred = w * x
        loss = (y_pred - y_target) ** 2

        print(f"  1. 前向计算: 预测值 = {y_pred.item():.4f}, Loss = {loss.item():.4f}")

        # ==========================================
        # 步骤 2: 反向传播 (Backward)
        # 这一行代码替代了你之前手写的链式法则计算 (grad_pred * grad_w_local)
        # 它会自动计算 dLoss/dw 并把结果存入 w.grad 中
        # ==========================================
        loss.backward()

        print(f"  2. 反向传播: PyTorch自动计算的梯度 w.grad = {w.grad.item():.4f}")

        # ==========================================
        # 步骤 3: 梯度更新 (Update)
        # ==========================================
        # with torch.no_grad(): 意思是 "接下来的计算不要追踪梯度"
        # 因为我们只是想修改 w 的数值，如果不加这句，PyTorch 会以为修改 w 也是模型计算的一部分
        with torch.no_grad():
            w -= learning_rate * w.grad  # 等同于 w = w - lr * grad

            # 【关键点】清空梯度！
            # 在你手写的代码里，grad 是每次重新算的变量。
            # 但 PyTorch 默认会把梯度累加（Accumulate）。如果不清零，下一次梯度就是“这次+上次”的结果。
            w.grad.zero_()

        print(f"  3. 参数更新: 新 w = {w.item():.4f}")
        print("-" * 30)

    print(f"\n最终结果: w = {w.item():.4f}")

if __name__ == "__main__":
    pytorch_demo()