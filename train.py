import time

import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import torch.nn as nn
import copy
import numpy as np
import matplotlib.pyplot as plt
from LeNet import LeNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_val_data_process():
    data = FashionMNIST(root="./data",
                        train=True,
                        transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                        download=True)

    train_data, val_data = Data.random_split(data, [round(0.8 * len(data)), round(0.2 * len(data))])
    train_loader = Data.DataLoader(dataset=train_data,
                                   batch_size=128,
                                   shuffle=True,
                                   num_workers=8)

    val_loader = Data.DataLoader(dataset=val_data,
                                 batch_size=128,
                                 shuffle=True,
                                 num_workers=8)

    return train_loader, val_loader


def train_model_process(model, loader, epochs, lr=1e-3, device=None):

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()  # 进入训练模式
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # 1. 清零梯度
            optimizer.zero_grad()

            # 2. 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 3. 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Loss: {epoch_loss:.4f} "
              f"Acc: {epoch_acc:.4f}")

    return model


def evaluate_model(model, val_loader):
    model.eval()  # 进入评估模式，关闭 dropout / batchnorm 更新

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 验证不需要梯度
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total

    print(f"Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
    return val_loss, val_acc

if __name__ == "__main__":
    loader_train, loader_val = train_val_data_process()
    le_net = train_model_process(LeNet(), loader_train, epochs=50)
    evaluate_model(le_net, loader_val)

    torch.save(le_net.state_dict(), "lenet_state_dict.pth")