import os
import torch
import torch.nn as nn
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
    linear = nn.Linear(784, 10)
    nn.init.normal_(linear.weight, mean=0, std=0.01)
    nn.init.constant_(linear.bias, 0)
    net = nn.Sequential(nn.Flatten(), linear)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1)

    epochs = 10

    for epoch in range(epochs):
        net.train()
        total_loss, total_samples = 0.0, 0
        for batchx, batchy in train_loader:
            optimizer.zero_grad()
            batch_loss = loss(net(batchx), batchy)
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item() * batchy.numel()
            total_samples += batchy.numel()

        avg_loss = total_loss / total_samples
        accuracy = evaluate_accuracy(net, test_loader)
        print(f"Epoch {epoch + 1:02d}/{epochs} | Train Loss: {avg_loss:.4f} | Test Acc: {accuracy:.4f}")


if __name__ == '__main__':
    main()