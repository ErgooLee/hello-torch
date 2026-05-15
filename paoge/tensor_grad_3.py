import torch

# 参数
w = torch.tensor([[1.0]], requires_grad=True)  # shape [1,1]
optimizer = torch.optim.SGD([w], lr=0.1)

# 数据：4 个样本，每个样本 1 个特征
x = torch.tensor([[1.],
                  [2.],
                  [3.],
                  [4.]])  # shape [4,1]
y = torch.tensor([[2.],
                  [4.],
                  [6.],
                  [8.]])  # shape [4,1]

# 方法一：一次性 batch
optimizer.zero_grad()
loss = ((w * x - y) ** 2).mean()
loss.backward()
print("一次性 batch 的梯度:", w.grad)
optimizer.step()
print(f"w={w.item()}")

# 方法二：梯度累积
w = torch.tensor([[1.0]], requires_grad=True)  # 重新初始化参数
optimizer = torch.optim.SGD([w], lr=0.1)

optimizer.zero_grad()
batch_size = x.shape[0]

for i in range(batch_size):
    xi = x[i:i+1]
    print(f"xi={xi}")# 保持 batch 维度 [1,1]
    yi = y[i:i+1]
    loss = ((w * xi - yi) ** 2)
    loss.backward()

w.grad /= 4
print("梯度累积的梯度:", w.grad)
optimizer.step()
print(f"w={w.item()}")