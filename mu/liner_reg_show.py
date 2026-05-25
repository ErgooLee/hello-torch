import torch
import matplotlib.pyplot as plt


def synthetic_data(w, b, num_examples):
    """y = Xw +b + c"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.mv(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# 创建 3D 图形
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# features[:, 0] 是 X1, features[:, 1] 是 X2
ax.scatter(features[:, 0].numpy(),
           features[:, 1].numpy(),
           labels.numpy(),
           s=5, color='royalblue', alpha=0.6, label='Data points')

# 设置坐标轴标签
ax.set_xlabel('X1 (Feature 1)')
ax.set_ylabel('X2 (Feature 2)')
ax.set_zlabel('y (Label)')

# 标题 (使用索引获取 w 的每个元素)
ax.set_title(f'Synthetic Data (y = {true_w[0]:.1f}*X1 + {true_w[1]:.1f}*X2 + {true_b})')
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()

# ==================== 第二张图：2D 投影 (只看 X1 与 y) ====================
plt.figure(figsize=(8, 5))

# 横坐标只画第一个特征 X1 (features[:, 0])
plt.scatter(features[:, 0].numpy(), labels.numpy(), s=15, color='royalblue', alpha=0.8, label='Data points')

# 绘制 X1 的理论趋势线 (y = 2 * X1 + 4.2)
# 注意：这里使用 true_w[0].item() 代替了 true_w.item()
x_line = torch.linspace(-3, 3, 100)
y_line = x_line * true_w[0].item() + true_b
plt.plot(x_line.numpy(), y_line.numpy(), color='firebrick', linewidth=2, label='Theoretical line for X1')

plt.xlabel('X1 (Feature 1)')
plt.ylabel('y (Label)')
# 标题中的 true_w.item() 也修改为 true_w[0].item()
plt.title(f'2D Projection: X1 vs y (Slope = {true_w[0].item():.1f})')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()
