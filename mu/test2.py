import random
import torch


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        start = i
        end = min(i + batch_size, num_examples)
        batch_indices = indices[start:end]
        yield features[batch_indices], labels[batch_indices]


def data_generator(w, b, num_examples):
    features = torch.normal(0, 1.0, (num_examples, len(w)))
    labels = torch.mm(features, w) + b
    labels += torch.normal(0, 0.01, labels.shape)
    return features, labels.reshape(-1, 1)


def grand_loss(y_hat, y):
    return ((y_hat - y) ** 2 / 2).mean()


def line_reg(features, w, b):
    return torch.mm(features, w) + b


def sgd(params, lr):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()


true_w = torch.tensor([3.7, 9.8]).reshape(-1, 1)
true_b = 4.3
num_examples = 1000
batch_size = 10

net = line_reg
loss = grand_loss
lr = 0.03

epoch = 3

features, labels = data_generator(true_w, true_b, num_examples)

w = torch.tensor([1.3, 2.1]).reshape(-1, 1).requires_grad_(True)
b = torch.tensor([1.3], requires_grad=True)

for epoch in range(epoch):
    for batch_feature, batch_label in data_iter(batch_size, features, labels):
        y_hat = net(batch_feature, w, b)
        loss_value = loss(y_hat, batch_label)
        loss_value.backward()
        sgd([w, b], lr)

    with torch.no_grad():
        epoch_loss = grand_loss(net(features, w, b), labels).mean()
        print(f"epoch {epoch}, loss {loss_value}")

print(f"true w = {true_w.tolist()}, true b = {true_b}")
print(f"     w = {w.tolist()},     b = {b.item()}")
