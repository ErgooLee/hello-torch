import torch
from torch import nn, device
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils import data
from torchinfo import summary


def get_device() -> device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Running on {device}")
    return device


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # 层1：原版 11x11 s4，适配 28x28 改为 3x3 s1
            nn.Conv2d(1, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 28 -> 14

            # 层2：原版 5x5 p2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 14 -> 7

            # 层3：3x3 p1
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),

            # 层4：3x3 p1
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),

            # 层5：3x3 p1
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 7 -> 3
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


def load_data(batch_size):
    trans = transforms.Compose([transforms.Resize(28), transforms.ToTensor()])
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


def show_AlexNet():
    device = get_device()

    train_loader, test_loader = load_data(batch_size=128)

    model = AlexNet().to(device)

    summary(model, input_size=(1, 1, 28, 28), col_names=["input_size", "output_size", "num_params", "kernel_size"])

    # train(model, train_loader, test_loader, epochs=10, lr=1e-3, device=device)


if __name__ == '__main__':
    show_AlexNet()
