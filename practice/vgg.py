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


def vgg_block(num_convs: int, in_channels: int, out_channels: int) -> nn.Sequential:
    """构建一个 VGG 块：包含 num_convs 个 3x3 卷积层 + ReLU，末尾接一个 2x2 最大池化层 (stride=2)。"""
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class VGG(nn.Module):
    """
    VGG 网络模型
    :param conv_arch: 卷积架构配置列表，每个元素为 (num_convs, out_channels)
    :param in_channels: 输入图像通道数 (FashionMNIST 为 1, RGB 为 3)
    :param num_classes: 分类类别数 (默认为 10)
    :param fc_hidden_dim: 全连接隐藏层维度 (默认为 4096，轻量化时可缩小)
    """
    def __init__(self, conv_arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)),
                 in_channels: int = 1, num_classes: int = 10, fc_hidden_dim: int = 4096):
        super().__init__()
        self.conv_arch = conv_arch

        # 1. 卷积块堆叠部分
        conv_blocks = []
        current_in_channels = in_channels
        for num_convs, out_channels in conv_arch:
            conv_blocks.append(vgg_block(num_convs, current_in_channels, out_channels))
            current_in_channels = out_channels
        self.conv = nn.Sequential(*conv_blocks)

        # 2. 自适应池化层，保证任意输入分辨率经过 5 个 block 后输出为 7x7 (与标准 VGG 保持一致)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # 3. 全连接分类头
        last_out_channels = conv_arch[-1][1]
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(last_out_channels * 7 * 7, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.adaptive_pool(x)
        x = self.fc(x)
        return x


def vgg11(in_channels: int = 1, num_classes: int = 10, ratio: int = 1) -> VGG:
    """
    生成 VGG-11 模型
    :param ratio: 通道数缩放因子 (如 ratio=4 可用于快速实验与轻量训练)
    """
    # VGG-11 基础架构：5 个 block，卷积层数分别为 [1, 1, 2, 2, 2]
    arch = ((1, 64 // ratio), (1, 128 // ratio), (2, 256 // ratio), (2, 512 // ratio), (2, 512 // ratio))
    fc_dim = 4096 // ratio if ratio > 1 else 4096
    return VGG(conv_arch=arch, in_channels=in_channels, num_classes=num_classes, fc_hidden_dim=fc_dim)


def load_data(batch_size: int, resize: int = 224):
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


def show_VGG():
    device = get_device()

    # 默认使用 ratio=4 的轻量化 VGG-11 打印数据流，便于查看和快速训练
    model = vgg11(in_channels=1, num_classes=10, ratio=4).to(device)

    print_model_flow(model, input_size=(1, 1, 224, 224))

    # train_loader, test_loader = load_data(batch_size=128, resize=224)
    # train(model, train_loader, test_loader, epochs=10, lr=1e-3, device=device)


if __name__ == '__main__':
    show_VGG()
