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


class Inception(nn.Module):
    """
    GoogLeNet 核心组件：Inception 块
    精髓 1：多尺度并行卷积 (1x1, 3x3, 5x5) 提取不同感受野的特征并融合；
    精髓 2：利用 1x1 卷积进行通道降维 (Bottleneck 设计)，大幅降低参数量和计算量。
    """
    def __init__(self, in_channels: int, c1: int, c2: tuple[int, int], c3: tuple[int, int], c4: int):
        super().__init__()
        # 分支 1: 单 1x1 卷积
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.ReLU()
        )
        # 分支 2: 1x1 降维 + 3x3 卷积 (padding=1 保持尺寸)
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
            nn.ReLU()
        )
        # 分支 3: 1x1 降维 + 5x5 卷积 (padding=2 保持尺寸)
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2),
            nn.ReLU()
        )
        # 分支 4: 3x3 最大池化 (stride=1, padding=1 保持尺寸) + 1x1 卷积调整通道
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        p1 = self.b1(x)
        p2 = self.b2(x)
        p3 = self.b3(x)
        p4 = self.b4(x)
        # 在通道维度拼接各分支输出
        return torch.cat([p1, p2, p3, p4], dim=1)


class GoogLeNet(nn.Module):
    """
    GoogLeNet (Inception-v1) 简化版：
    精髓 3：模块化 5 阶段架构 (Stage 1~5)；
    精髓 4：使用全局平均池化 (GAP) 替代庞大的全连接层，参数量远小于 AlexNet 和 VGG。
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10, ratio: int = 1):
        super().__init__()
        def scale(c):
            return max(1, c // ratio)

        def scale_tuple(t):
            return (scale(t[0]), scale(t[1]))

        # Stage 1: Stem 基础特征提取 (下采样)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, scale(64), kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Stage 2: 进一步提特征 + 降维
        self.stage2 = nn.Sequential(
            nn.Conv2d(scale(64), scale(64), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(scale(64), scale(192), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Stage 3: Inception 块堆叠 (3a, 3b) + 下采样
        in_c3 = scale(192)
        inc3a = Inception(in_c3, scale(64), scale_tuple((96, 128)), scale_tuple((16, 32)), scale(32))
        c3a_out = scale(64) + scale(128) + scale(32) + scale(32)

        inc3b = Inception(c3a_out, scale(128), scale_tuple((128, 192)), scale_tuple((32, 96)), scale(64))
        c3b_out = scale(128) + scale(192) + scale(96) + scale(64)

        self.stage3 = nn.Sequential(
            inc3a,
            inc3b,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Stage 4: Inception 块堆叠 (4a ~ 4e) + 下采样
        inc4a = Inception(c3b_out, scale(192), scale_tuple((96, 208)), scale_tuple((16, 48)), scale(64))
        c4a_out = scale(192) + scale(208) + scale(48) + scale(64)

        inc4b = Inception(c4a_out, scale(160), scale_tuple((112, 224)), scale_tuple((24, 64)), scale(64))
        c4b_out = scale(160) + scale(224) + scale(64) + scale(64)

        inc4c = Inception(c4b_out, scale(128), scale_tuple((128, 256)), scale_tuple((24, 64)), scale(64))
        c4c_out = scale(128) + scale(256) + scale(64) + scale(64)

        inc4d = Inception(c4c_out, scale(112), scale_tuple((144, 288)), scale_tuple((32, 64)), scale(64))
        c4d_out = scale(112) + scale(288) + scale(64) + scale(64)

        inc4e = Inception(c4d_out, scale(256), scale_tuple((160, 320)), scale_tuple((32, 128)), scale(128))
        c4e_out = scale(256) + scale(320) + scale(128) + scale(128)

        self.stage4 = nn.Sequential(
            inc4a, inc4b, inc4c, inc4d, inc4e,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Stage 5: Inception 块堆叠 (5a, 5b) + 全局平均池化
        inc5a = Inception(c4e_out, scale(256), scale_tuple((160, 320)), scale_tuple((32, 128)), scale(128))
        c5a_out = scale(256) + scale(320) + scale(128) + scale(128)

        inc5b = Inception(c5a_out, scale(384), scale_tuple((192, 384)), scale_tuple((48, 128)), scale(128))
        c5b_out = scale(384) + scale(384) + scale(128) + scale(128)

        self.stage5 = nn.Sequential(
            inc5a,
            inc5b,
            nn.AdaptiveAvgPool2d((1, 1))                      # 全局平均池化 (GAP): [B, Cout, 1, 1]
        )

        # 分类器：仅需 1 个线性层
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(c5b_out, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
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


def show_GoogLeNet():
    device = get_device()

    model = GoogLeNet(in_channels=1, num_classes=10).to(device)

    # 打印数据流，输入大小设为 96x96 (或 224x224)
    print_model_flow(model, input_size=(1, 1, 96, 96))

    train_loader, test_loader = load_data(batch_size=128, resize=96)
    train(model, train_loader, test_loader, epochs=10, lr=1e-3, device=device)


if __name__ == '__main__':
    show_GoogLeNet()
