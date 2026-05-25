import torch

true_w = torch.tensor([2.3, 3.4]).reshape(2, -1)
true_b = 1.1

num_feature = 20
features = torch.normal(0, 1, (num_feature, 2))
y = torch.matmul(features, true_w) + true_b
c = torch.normal(0, 0.01, (num_feature, 1))
y += c

epochs = 50
lr = 0.1
w = torch.normal(0, 1, (2, 1), requires_grad=True)
b = torch.normal(0, 1, (1, 1), requires_grad=True)

for epoch in range(epochs):
    y_hat = torch.matmul(features, w) + b
    loss = ((y_hat - y) ** 2 / 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
    if (epoch + 1) % 10 == 0:
        print(f"epoch {epoch + 1}, loss {loss.item():.4f}")

print("\n--- 训练结束 ---")
print(f"真实 w:\n{true_w.tolist()}")
print(f"学到的 w:\n{w.tolist()}")
print(f"真实 b: {true_b}")
print(f"学到的 b: {b.item():.4f}")