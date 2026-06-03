
import torch
from my_lib import synthetic_data
from my_lib import data_iter
from my_lib import liner_reg
from my_lib import squared_loss
from my_lib import sgd

def main():
    true_w = torch.tensor([[2], [-3.4]])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    batch_size = 10
    for x, y in data_iter(batch_size, features, labels):
        print(x, '\n', y)
        break

    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

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


if __name__ == "__main__":
    main()
