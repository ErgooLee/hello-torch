import torch
from torch.utils import data
from torch import nn

def mock_data(w, b, sample_num):
    x = torch.normal(0, 1, (sample_num, len(w)))
    y = x @ w + b
    y += torch.normal(0, 0.03, y.shape)
    return x, y.reshape(-1, 1)


def main():
    true_w = torch.tensor([103, -89], dtype=torch.float).reshape(-1, 1)
    true_b = -3.9
    sample_num = 1000
    features, labels = mock_data(true_w, true_b, sample_num)
    dataset = data.TensorDataset(*(features, labels))

    epoch = 10

    net = nn.Sequential(nn.Linear(2, 1))
    nn.init.normal_(net[0].weight, mean=0.0, std=0.01)
    nn.init.constant_(net[0].bias, val=0.0)

    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.03)

    batch_size = 64

    for epoch in range(epoch):
        for batch_x, batch_y in data.DataLoader(dataset, batch_size=batch_size, shuffle=True):
            batch_loss = loss(net(batch_x), batch_y)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        with torch.no_grad():
            epoch_loss = loss(net(features), labels)
            print(f'Epoch {epoch}, Loss: {epoch_loss.item()}')
    print(f"true w: {true_w.squeeze().tolist()}, true b: {true_b:.4f}")
    print(f"net  w: {net[0].weight.squeeze().tolist()}, net b: {net[0].bias.item():.4f}")


if __name__ == "__main__":
    main()
