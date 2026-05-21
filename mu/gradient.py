import torch

x = torch.arange(4.0)
print(x)
print(x.shape)
print(x.dtype)

x.requires_grad_(True)

print(x.grad)

y = 2 * torch.dot(x, x)

y.backward()

print(x.grad)

# x.grad.zero_()
# print(x.grad)
y = x.sum()
y.backward()
print(x.grad)

x.grad.zero_()

x = torch.arange(4.0, requires_grad=True)
y = x * x
print(y)
y.backward(torch.ones_like(y))
print(x.grad)

print("test3-----")
x = torch.arange(4.0, requires_grad=True)
y = x * x
u = y.detach()
z = u * x
z.backward(torch.ones_like(z))
print(x.grad)

print("test4-----")
x = torch.arange(4.0, requires_grad=True)
y = x * x
u = y
z = u * x
z.backward(torch.ones_like(z))
print(x.grad)

print("test5-----")


def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad == d / a)
