import torch
import numpy as np

# 1. 创建 Tensor (像捏泥巴一样创造它)
# 最常用的几种创建方式：
print("创建 Tensor----------------")
# 1. 直接从列表创建 (最直观)
x = torch.tensor([1, 2, 3])      # 1维向量
print(f"x={x}")
y = torch.tensor([[1, 2], [3, 4]]) # 2维矩阵
print(f"y={y}")

# 2. 创建全0、全1、随机数 (初始化参数时常用)
a = torch.zeros(2, 3)            # 2行3列的全0矩阵
print(f"a={a}")
b = torch.ones(2, 3)             # 2行3列的全1矩阵
print(f"b={b}")
c = torch.randn(2, 3)            # 标准正态分布随机数 (均值0，方差1)
print(f"c={c}")
d = torch.rand(2, 3)             # 0到1之间的均匀分布随机数
print(f"d={d}")
# 3. 像谁一样 (非常实用！)
# 创建一个形状、设备都和 a 一样的全0矩阵
e = torch.zeros_like(a)
print(f"e={e}")
# 4. 指定类型 (重要)
# 深度学习通常使用 float32，整数索引使用 long (int64)
f = torch.tensor([1, 2], dtype=torch.float32)
print(f"f={f}")

# 2. 查看属性 (查户口)
# 拿到一个 Tensor，你首先要确认它的“三维”信息。
print("查看属性----------------")
print(f"shape_e={e.shape},dtype={e.dtype},device={e.device}")

# 3. 维度变换 (深度学习中最频繁的操作)
# 由于神经网络对输入形状非常敏感，你经常需要把形状揉圆搓扁。
print("维度变换----------------")
x = torch.rand(2, 3, 4)  # 假设这是一个 [Batch, Height, Width]
print(f"x={x}")
# 1. view / reshape: 改变形状 (保持总元素个数不变)
# 把 (2, 3, 4) 变成 (2, 12)
y = x.view(2, 12)
print(f"y={y}")
z = x.reshape(2, -1)    # -1 表示“自动推算”，这里会自动算出 12
print(f"z={z}")

# 2. squeeze / unsqueeze: 增减维度
# 增加维度 (通常为了凑 Batch 维度)
a = torch.rand(2, 2)
print(f"a={a}")
b = a.unsqueeze(0)      # 在第0维增加，变成 (1, 2, 2)
print(f"b={b}")
# 减少维度 (去掉只有 1 的维度)
c = b.squeeze()         # 变回 (2, 2)
print(f"c={c}")
# 3. permute / transpose: 交换维度顺序
# 比如把图片从 [H, W, C] 变成 [C, H, W]
img = torch.rand(3, 4, 2)
print(f"img={img}")
img_new = img.permute(2, 0, 1)  # 变成 (2, 3, 4)
print(f"img_new={img_new}")

print("4、数学运算----------------")
# 数学运算 (加减乘除与矩阵乘法)
# 注意区分 点乘 和 矩阵乘法。
x = torch.tensor([[1., 2], [3, 4]])
y = torch.tensor([[2., 2], [2, 2]])

# 1. 对应元素运算 (Element-wise)
print(f"x+y={x + y}")    # 加
print(f"x*y={x * y}")    # 对应位置相乘 (不是矩阵乘法！) -> [[2, 4], [6, 8]]

# 2. 矩阵乘法 (Matrix Multiplication) - 核心！
# 也就是线性代数里的行乘列
print(f"x*y={x @ y}")             # 写法1 (推荐，Python 3.5+)
print(f"x matmul y={torch.matmul(x, y)}") # 写法2

# 3. 聚合操作
print(f"sum={x.sum()}")           # 求和
print(f"mean={x.mean()}")          # 求平均
print(f"max={x.max()}")           # 求最大值
print(f"argmax={x.argmax()}")        # 求最大值所在的索引 (分类任务常用)

print("5、取值与切片----------------")
# 取值与切片 (和 NumPy 一模一样)
x = torch.arange(10)
print(f"x={x}")
x = x.reshape(2, 5)
print(f"x={x}")
# [[0, 1, 2, 3, 4],
#  [5, 6, 7, 8, 9]]

# 切片
print(x[:, 1:3])  # 取所有行，第1到2列 -> [[1, 2], [6, 7]]

# 取单值 (特别注意！)
loss = torch.tensor([0.0523])
# print(loss) 得到的是 Tensor 对象
val = loss.item() # 得到标准的 Python float 数字 (0.0523)
# 只有包含一个元素的 tensor 才能用 .item()

# 6. CPU 与 GPU 互通 (搬家)
print("6. CPU 与 GPU 互通 (搬家)----------------")
x = torch.tensor([1, 2, 3])

if torch.cuda.is_available():
    # 搬去 GPU
    x_gpu = x.to('cuda')  # 或者 x.cuda()
    print(x_gpu.device)  # device='cuda:0'

    # 运算 (必须都在同一个设备上)
    # y = x + x_gpu  # 报错！不能 CPU 加 GPU
    y = x_gpu + x_gpu  # 成功，结果也在 GPU 上

    # 搬回 CPU (比如要用 matplotlib 画图，或者转 numpy)
    x_cpu = x_gpu.to('cpu')  # 或者 x_gpu.cpu()

# 7.与 NumPy 互转
print("7.与 NumPy 互转----------------")
# Tensor -> NumPy
t = torch.tensor([1., 2.])
n = t.numpy()  # 注意：如果 t 在 GPU 上，要先 .cpu() 才能转 numpy

# NumPy -> Tensor
n = np.array([1., 2.])
t = torch.from_numpy(n)