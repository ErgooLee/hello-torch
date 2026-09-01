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


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(84, 10),
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
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X.size(0)
            correct += model(X).argmax(dim=1).eq(y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{epochs}  "
              f"loss {running_loss / total:.4f}  "
              f"train_acc {train_acc:.4f}  "
              f"test_acc {test_acc:.4f}")


def show_LeNet():
    device = get_device()

    train_loader, test_loader = load_data(batch_size=128)

    model = LeNet().to(device)

    summary(model, input_size=(1, 1, 28, 28), col_names=["input_size", "output_size", "num_params", "kernel_size"])

    # train(model, train_loader, test_loader, epochs=10, lr=1e-3, device=device)


if __name__ == '__main__':
    show_LeNet()
