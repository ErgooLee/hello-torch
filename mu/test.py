import torch

true_w = torch.tensor([[3.9], [0.9]])
true_b = 100

num_feature = 100
features = torch.normal(0, 1, (num_feature, 2))
y = torch.mm(features, true_w) + true_b
y += torch.normal(0, 0.01, (num_feature, 1))

w = torch.tensor([[7.3], [1.9]], requires_grad=True)
b = torch.tensor(3.8, requires_grad=True)
lr = 0.4

epochs = 30

for epoch in range(epochs):
    y_hat = torch.mm(features, w) + b
    loss = ((y_hat - y) ** 2 / 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
    if epoch % 10 == 0:
        print(f'epoch {epoch}, loss {loss.item():.4f}')

print(f"true w: {true_w}, true b: {true_b}")
print(f"w: {w}, b: {b}")
