import torch
from torch import nn
from torch.utils import data


def mock_data(w, b, sample_num):
    x = torch.normal(0, 1, (sample_num, len(w)))
    y = x @ w + b
    y += torch.normal(0, 0.03, y.shape)
    return x, y.reshape(-1, 1)


def main():
    first_layer = nn.Linear(2, 1)
    net = nn.Sequential(first_layer)
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    nn.init.normal_(first_layer.weight, mean=0.0, std=0.01)
    nn.init.constant_(first_layer.bias, val=0.0)

    true_w = torch.tensor([[100], [-32]], dtype=torch.float)
    true_b = 3.9

    batch_size = 64
    sample_num = 1000

    features, labels = mock_data(true_w, true_b, sample_num)

    dataset = data.TensorDataset(*(features, labels))

    epochs = 10
    for epoch in range(epochs):
        for batch_x, batch_y in data.DataLoader(dataset, batch_size, shuffle=True):
            batch_loss = loss(net(batch_x), batch_y)
            trainer.zero_grad()
            batch_loss.backward()
            trainer.step()
        with torch.no_grad():
            epoch_loss = loss(net(features), labels)
            print(f"epoch {epoch}, loss {epoch_loss}")
    print(f"w = {first_layer.weight.data.squeeze().tolist()}, b = {first_layer.bias.data.item()}")
    print(f"true_w = {true_w.squeeze().tolist()}, true_b = {true_b:.4f}")


if __name__ == "__main__":
    main()
