import os
import torch
import torch.nn as nn
from torch.nn import ReLU
from torchvision import datasets, transforms
from torch.utils import data
from torch import optim


def correct_count(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    return (y_hat == y).sum().item()


def evaluate_accuracy(net, data_loader):
    net.eval()
    right_count = 0
    with torch.no_grad():
        for x, y in data_loader:
            right_count += correct_count(net(x), y)
    return right_count / len(data_loader.dataset)


def main():
    batch_size = 64
    transform = transforms.Compose([transforms.ToTensor()])

    # 路径自适应
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
    train_set = datasets.FashionMNIST(root=data_dir, train=True, download=False, transform=transform)
    test_set = datasets.FashionMNIST(root=data_dir, train=False, transform=transform)

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # 模型搭建与初始化
    first_layer = nn.Linear(784, 256)
    second_layer = nn.Linear(256, 10)

    # 2. 故意将所有层权重全部初始化为 0（陷阱）
    # nn.init.constant_(first_layer.weight, 0.0)
    # nn.init.constant_(first_layer.bias, 0.0)
    # nn.init.constant_(second_layer.weight, 0.0)
    # nn.init.constant_(second_layer.bias, 0.0)

    net = nn.Sequential(nn.Flatten(), first_layer, nn.ReLU(), second_layer)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1)

    epochs = 10

    for epoch in range(epochs):
        net.train()
        right_count_train = 0
        total_loss, total_samples = 0.0, 0
        for batchx, batchy in train_loader:
            optimizer.zero_grad()
            y_hat = net(batchx)
            batch_loss = loss(y_hat, batchy)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item() * batchy.numel()
            total_samples += batchy.numel()
            right_count_train += correct_count(y_hat, batchy)

        avg_loss = total_loss / total_samples
        accuracy_train = right_count_train / total_samples
        accuracy = evaluate_accuracy(net, test_loader)
        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f} | Train Acc :{accuracy_train:.4f} Test Acc: {accuracy:.4f}")


if __name__ == '__main__':
    main()
