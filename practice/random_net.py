import torch
import torch.nn as nn
import torch.optim as optim


class OverfitNet(nn.Module):

  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(20, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),  # 二分类输出 0~1 概率
    )

  def forward(self, x):
    return self.net(x)

def test_nothing():
    # 1. 生成毫无关系的“纯随机噪声”数据
    torch.manual_seed(42)

    # 训练集：500 个样本，每个样本 20 个随机特征；标签也是随机 0 或 1
    X_train = torch.randn(500, 20)
    y_train = torch.randint(0, 2, (500,)).float().unsqueeze(1)

    # 测试集：另外 200 个独立的纯随机样本
    X_test = torch.randn(200, 20)
    y_test = torch.randint(0, 2, (200,)).float().unsqueeze(1)

    model = OverfitNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 3. 开始训练
    print("开始训练纯随机数据...\n")
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            # 计算训练集准确率
            train_acc = ((preds >= 0.5) == y_train).float().mean().item()

            # 计算测试集准确率
            model.eval()
            with torch.no_grad():
                test_preds = model(X_test)
                test_loss = criterion(test_preds, y_test)
                test_acc = ((test_preds >= 0.5) == y_test).float().mean().item()

            print(
                f"Epoch [{epoch:3d}/200] | "
                f"训练集 Loss: {loss.item():.4f}, 准确率: {train_acc * 100:5.1f}% | "
                f"测试集 准确率: {test_acc * 100:5.1f}% (纯瞎猜水平)"
            )

if __name__ == '__main__':
    test_nothing()