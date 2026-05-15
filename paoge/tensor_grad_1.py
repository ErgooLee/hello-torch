import torch

w = torch.tensor(1.0, requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.1)

x = torch.tensor([1., 2., 3., 4.])
y = 2 * x   # 真值

optimizer.zero_grad()
loss = ((w*x - y)**2).mean()
loss.backward()
print(w.grad)
optimizer.step()
print(w)

