import torch
import random


def mock_data(w, b, data_num):
    x = torch.normal(0, 1, (data_num, len(w)))
    y = x @ w + b
    y += torch.normal(0, 0.03, y.shape)
    return x, y


def generate_batch(batch_size, x, y):
    data_num = len(x)
    indices = list(range(data_num))
    random.shuffle(indices)
    for index in range(0, data_num, batch_size):
        batch_indices = indices[index:min(index + batch_size, data_num)]
        yield x[batch_indices], y[batch_indices]


def linear_reg(x, w, b):
    return x @ w + b


def squared_loss(y_hat, y):
    return ((y_hat - y.reshape(y_hat.shape)) ** 2 / 2).mean()


def update_params(params, lr):
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad
            p.grad.zero_()


def main():
    true_w = torch.tensor([8.3, -7.9], dtype=torch.float).reshape(-1, 1)
    true_b = torch.tensor([-98], dtype=torch.float)
    data_num = 1000
    features, labels = mock_data(true_w, true_b, data_num)

    epochs = 10
    batch_size = 64
    lr = 0.05

    w = torch.normal(0, 0.01, (len(true_w), 1), dtype=torch.float, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float, requires_grad=True)

    print(f"w: {w.squeeze().tolist()}, b: {b.item():.4f}")

    for epoch in range(epochs):
        for batch_x, batch_y in generate_batch(batch_size, features, labels):
            batch_y_hat = linear_reg(batch_x, w, b)
            loss = squared_loss(batch_y_hat, batch_y)
            loss.backward()
            update_params([w, b], lr)
        with torch.no_grad():
            total_loss = squared_loss(linear_reg(features, w, b), labels)
            print(f"epoch {epoch + 1}, loss: {total_loss.item():.4f}")

    print(f"w: {w.squeeze().tolist()}, b: {b.item():.4f}")
    print(f"real_w: {true_w.squeeze().tolist()}, b: {true_b.item():.4f}")


if __name__ == '__main__':
    main()
