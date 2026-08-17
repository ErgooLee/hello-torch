import torch
import random


def generate_data(w, b, example_num):
    x = torch.normal(0, 1, (example_num, len(w)))
    y = x @ w + b
    y += torch.normal(0, 0.03, y.shape)
    return x, y.reshape(-1, 1)


def get_patch(batch_size, features, labels):
    example_num = len(labels)
    index_array = list(range(example_num))
    random.shuffle(index_array)
    for i in range(0, example_num, batch_size):
        sub_index = index_array[i:min(i + batch_size, example_num)]
        yield features[sub_index], labels[sub_index]


def line_reg(x, w, b):
    return x @ w + b


def squared_loss(y_hat, y):
    return ((y_hat - y.reshape(-1, 1)) ** 2 / 2).mean()


def update_params(params, lr):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()


def main():
    true_w = torch.tensor([[98], [-3.6]], dtype=torch.float)
    true_b = -3.7
    features, labels = generate_data(true_w, true_b, 1000)

    w = torch.tensor([9, 100], dtype=torch.float).reshape(-1, 1).requires_grad_()
    b = torch.tensor([-100], dtype=torch.float, requires_grad=True)
    lr = 0.03

    epoches = 20
    batch_size = 10
    for epoch in range(epoches):
        for batch_feature, batch_label in get_patch(batch_size, features, labels):
            y_hat = line_reg(batch_feature, w, b)
            loss = squared_loss(y_hat, batch_label)
            loss.backward()
            update_params((w, b), lr)
        with torch.no_grad():
            loss_epoch = squared_loss(line_reg(features, w, b), labels)
            print("epoch %d, loss %f" % (epoch, loss_epoch.item()))
    print(f"w = {w.squeeze().tolist()}, b = {b.item():.4f}")
    print(f"true_w = {true_w.squeeze().tolist()}, true_b = {true_b:.4f}")


if __name__ == "__main__":
    main()
