import random

import torch


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i:min(i + batch_size, num_examples)]
        yield features[batch_indices], labels[batch_indices]


def synthetic_data(w, b, num_examples):
    """y = Xw +b + c"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.mv(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

batch_size = 10
for x, y in data_iter(batch_size, features, labels):
    print(x, '\n', y)
    break

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


def liner_reg(X, w, b):
    return torch.mm(X, w) + b


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()


lr = 0.03
num_epochs = 3
net = liner_reg
loss = squared_loss

for epoch in range(num_epochs):
    for x, y in data_iter(batch_size, features, labels):
        l = loss(net(x, w, b), y).mean()
        l.backward()
        sgd([w, b], lr)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels).mean()
        print(f'epoch {epoch + 1}, train loss: {float(train_l.mean()):.4f}')

print(f"w: {w}, b: {b}")
