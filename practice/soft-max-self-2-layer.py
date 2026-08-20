import math

import torch
from torchvision import datasets, transforms
from torch.utils import data


# 64 * 10 64个样本，每个样本有10个输出，所以
def softmax(x):
    exp_x = torch.exp(x)
    partition = exp_x.sum(dim=1, keepdim=True)
    return exp_x / partition


def linear_layer(x, w, b):
    # 这里能将64*1*28*28展平为x*784=64*784
    return x.reshape(-1, w.shape[0]) @ w + b


def relu(x):
    return torch.maximum(x, torch.zeros_like(x))


def two_layers_net(x, w1, b1, w2, b2):
    y_hat_1 = relu(linear_layer(x, w1, b1))
    y_hat_2 = linear_layer(y_hat_1, w2, b2)
    return softmax(y_hat_2)


def test_softmax():
    x = torch.normal(0, 1, (8, 5), dtype=torch.float32)
    soft_x = softmax(x)
    print(f"x={x} \n soft{soft_x} \n sum1={soft_x.sum(1)}")


# -p*log(q)
def across_loss(y_hat, y):
    cols = range(len(y_hat))
    return -torch.log(y_hat[cols, y]).mean()


def test_across_loss():
    y_hat = torch.tensor([[0.1, 0.2, 0.7], [0.2, 0.2, 0.6]], dtype=torch.float32)
    y = torch.tensor([0, 2], dtype=torch.int)
    loss = across_loss(y_hat, y)
    real_loss = torch.tensor([-math.log(0.1), -math.log(0.6)]).mean()
    print(f"loss={loss} real_loss={real_loss}")


def update(params, lr):
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad
            p.grad.zero_()


def correct_count(y_hat, y):
    # 如果是概率分布，取最大概率的类别
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)

    # 直接比较（PyTorch 自动处理类型）
    correct_num = (y_hat == y).sum().item()

    return correct_num


def test_super_index():
    y = torch.tensor([0, 2])
    y_hat = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.2, 0.5]])
    print(f"rate = {y_hat[[0, 1], y]}")
    print(f"rate = {y_hat[:, y]}")

    y_hat = torch.arange(12).reshape(3, 4)
    idx1 = torch.tensor([[0, 1], [1, 2]])  # 2×2
    idx2 = torch.tensor([[0, 1], [1, 3]])  # 2×2

    result = y_hat[idx1, idx2]
    print(f"result={result}")


def main():
    transform = transforms.Compose([transforms.ToTensor()])

    # 不要用下面的这种方式做展平，不要一开始就让二维信息丢了。
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    train_set = datasets.FashionMNIST('../data', train=True, transform=transform, download=False)
    test_set = datasets.FashionMNIST('../data', train=False, transform=transform, download=False)
    print(f"train_set={len(train_set)}, test_set={len(test_set)}")

    batch_size = 64

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    num_input = 28 * 28
    num_hidden = 256
    num_output = 10

    w1 = torch.normal(0, 0.01, (num_input, num_hidden), dtype=torch.float32, requires_grad=True)
    b1 = torch.zeros(num_hidden, dtype=torch.float32, requires_grad=True)

    w2 = torch.normal(0, 0.01, (num_hidden, num_output), dtype=torch.float32, requires_grad=True)
    b2 = torch.zeros(num_output, dtype=torch.float32, requires_grad=True)

    epochs = 10

    lr = 0.1

    # 2. 增加每轮平均 Loss 打印
    for epoch in range(epochs):
        total_loss, total_samples = 0.0, 0
        for batchx, batchy in train_loader:
            y_hat = two_layers_net(batchx, w1, b1, w2, b2)
            batch_loss = across_loss(y_hat, batchy)
            batch_loss.backward()
            update((w1, b1, w2, b2), lr)

            total_loss += batch_loss.item() * batchy.numel()
            total_samples += batchy.numel()

        with torch.no_grad():
            right_count = 0
            for batchx, batchy in test_loader:
                right_count += correct_count(two_layers_net(batchx, w1, b1, w2, b2), batchy)
            accuracy = right_count / len(test_loader.dataset)
            avg_loss = total_loss / total_samples
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f} | Test Acc: {accuracy:.4f}")


if __name__ == '__main__':
    main()
