import torch

w = torch.tensor(1.0, requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.1)

x = torch.tensor([1., 2., 3., 4.])
y = 2 * x

optimizer.zero_grad()
for i in range(4):
    loss = (w*x[i] - y[i])**2 / 4   # 注意这里除以 4
    loss.backward()

print(w.grad)   # tensor(-15.)
optimizer.step()

print(w)


