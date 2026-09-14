import torch
from torch import nn, device
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils import data
from practice.model_summary import print_model_flow


def get_device() -> device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Running on {device}")
    return device


class Residual(nn.Module):
    """
    ResNet 核心组件：残差块 (Residual Block)
    精髓 1：跨层快捷连接 (Skip Connection)，拟合残差映射 F(x) = H(x) - x，解决网络退化与梯度消失；
    精髓 2：当通道数改变或空间分辨率减半 (stride > 1) 时，旁路使用 1x1 卷积 + BatchNorm 调整维度；
    精髓 3：主路径特征与旁路特征相加 (F(x) + x) 后再进行最后的 ReLU 激活。
    """
    def __init__(self, in_channels: int, out_channels: int, use_1x1conv: bool = False, stride: int = 1):
        super().__init__()
        # 主路径第 1 个 3x3 卷积 (可负责降采样 stride)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 主路径第 2 个 3x3 卷积 (保持尺寸与通道)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 旁路快捷连接 (Shortcut / Identity Mapping)
        if use_1x1conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 核心加法：F(x) + x
        out += identity
        out = self.relu(out)
        return out


def resnet_block(in_channels: int, out_channels: int, num_residuals: int, first_block: bool = False) -> nn.Sequential:
    """构建一个 ResNet Stage：由多个残差块堆叠而成。"""
    layers = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # 非第一个 Stage 的首个 Block：下采样 (stride=2) 并用 1x1 卷积升维
            layers.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        elif i == 0 and first_block:
            # 第一个 Stage 的首个 Block：通道一致时无需 1x1 卷积；通道不一致时用 1x1 卷积转换
            use_1x1 = (in_channels != out_channels)
            layers.append(Residual(in_channels, out_channels, use_1x1conv=use_1x1, stride=1))
        else:
            # Stage 内后续 Block：输入输出通道和尺寸一致
            layers.append(Residual(out_channels, out_channels))
    return nn.Sequential(*layers)


class ResNet18(nn.Module):
    """
    ResNet-18 简化版：
    精髓 4：Stem 卷积预处理 -> 4 个 Stage 残差模块堆叠 -> 全局平均池化 (GAP) + 单层分类器。
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10, ratio: int = 1):
        super().__init__()
        def scale(c):
            return max(1, c // ratio)

        c1 = scale(64)
        c2 = scale(128)
        c3 = scale(256)
        c4 = scale(512)

        # Stem: 基础卷积层提取低阶特征
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 4 个 Stage 残差模块
        self.stage1 = resnet_block(c1, c1, num_residuals=2, first_block=True)
        self.stage2 = resnet_block(c1, c2, num_residuals=2)
        self.stage3 = resnet_block(c2, c3, num_residuals=2)
        self.stage4 = resnet_block(c3, c4, num_residuals=2)

        # 全局平均池化 (GAP) + 线性分类器
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c4, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = self.fc(x)
        return x


def load_data(batch_size: int, resize: int = 96):
    trans = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
    train_set = FashionMNIST(root="./data", train=True, transform=trans, download=True)
    test_set = FashionMNIST(root="./data", train=False, transform=trans, download=True)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)
    return correct / total


def train(model, train_loader, test_loader, epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X.size(0)
            correct += logits.argmax(dim=1).eq(y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{epochs}  "
              f"loss {running_loss / total:.4f}  "
              f"train_acc {train_acc:.4f}  "
              f"test_acc {test_acc:.4f}")


def show_ResNet():
    device = get_device()

    # 实例化 ResNet18 模型 (可通过 ratio 缩放通道数进行轻量化)
    model = ResNet18(in_channels=1, num_classes=10, ratio=1).to(device)

    # 打印数据流管道 (输入 96x96)
    print_model_flow(model, input_size=(1, 1, 96, 96))

    train_loader, test_loader = load_data(batch_size=128, resize=96)
    train(model, train_loader, test_loader, epochs=10, lr=1e-3, device=device)


if __name__ == '__main__':
    show_ResNet()
