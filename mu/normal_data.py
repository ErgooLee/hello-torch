import torch

x = torch.arange(12)
print(x)
print(x.shape)
print(x.numel())
print(x.reshape(3, 4))

print(torch.zeros(2, 3, 4))

print(torch.ones(2, 3, 4))

print(torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])

print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y)
print(x.exp())

x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((x, y), dim=0))
print(torch.cat((x, y), dim=1))

print(x == y)

print(x.sum())
print(x.min())
print(x.max())

x = torch.arange(3).reshape(3, 1)
y = torch.arange(2).reshape(1, 2)

print(x + y)

x = torch.arange(12)
print(x[-1])
print(x[1:8])

x = torch.arange(12).reshape(3, 4)
print(x[-1])
print(x[1:8])

x[2, 2] = 104
print(x[2, 2])

x[1, :] = 100
print(x)

x = torch.arange(12).reshape(3, 4)
y = torch.arange(3, 15).reshape(3, 4)
before = id(y)
y = y + x
print(id(y) == before)

z = torch.zeros_like(y)
before = id(z)
z[:] = x + y
print(id(z) == before)

before = id(x)
x += y
print(id(x) == before)

x = torch.arange(12).reshape(3, 4)
y = x.numpy()
print(type(x))
print(type(y))

a = torch.tensor([3.5])
print(a)
print(a.item())
print(float(a))
print(int(a))

