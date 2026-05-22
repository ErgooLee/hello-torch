import torch
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import webbrowser


def synthetic_data(w, b, num_examples):
    """y = Xw +b + c"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.mv(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)


# 1. 生成数据
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# ==================== 1. 使用 Plotly 绘制 3D 可交互图 ====================

# 将 PyTorch 数据转换为 Pandas DataFrame，以便 Plotly 读取
df = pd.DataFrame({
    'X1': features[:, 0].numpy(),
    'X2': features[:, 1].numpy(),
    'y': labels.numpy().flatten()  # 展开为一维数组
})

# 使用 Plotly Express 绘制 3D 散点图
fig = px.scatter_3d(
    df, 
    x='X1', 
    y='X2', 
    z='y',
    title=f'Interactive 3D Synthetic Data (y = {true_w[0]:.1f}*X1 + {true_w[1]:.1f}*X2 + {true_b})',
    labels={'X1': 'X1 (Feature 1)', 'X2': 'X2 (Feature 2)', 'y': 'y (Label)'}
)

# 调整点的大小和透明度，使其视觉效果更接近原本的设置
fig.update_traces(
    marker=dict(
        size=3,              # 对应原代码中的 s=5 左右
        color='royalblue',   # 颜色
        opacity=0.7          # 透明度
    )
)

# 显示交互图（在浏览器或 Jupyter Notebook 中可自由旋转、缩放、平移）
fig.write_html("plotly_3d_plot.html")
webbrowser.open_new_tab("plotly_3d_plot.html")


# ==================== 2. 保留 2D 投影 (只看 X1 与 y) ====================
plt.figure(figsize=(8, 5))

# 横坐标只画第一个特征 X1 (features[:, 0])
plt.scatter(features[:, 0].numpy(), labels.numpy(), s=15, color='royalblue', alpha=0.8, label='Data points')

# 绘制 X1 的理论趋势线 (y = 2 * X1 + 4.2)
x_line = torch.linspace(-3, 3, 100)
y_line = x_line * true_w[0].item() + true_b
plt.plot(x_line.numpy(), y_line.numpy(), color='firebrick', linewidth=2, label='Theoretical line for X1')

plt.xlabel('X1 (Feature 1)')
plt.ylabel('y (Label)')
plt.title(f'2D Projection: X1 vs y (Slope = {true_w[0].item():.1f})')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()