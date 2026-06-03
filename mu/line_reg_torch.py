import torch

from torch import nn
from my_lib import synthetic_data
from my_lib import load_data

def main():
    true_w = torch.tensor([[1.8], [9.6]])
    true_b = 3.9

    features, labels = synthetic_data(true_w, true_b, num_examples=1000)

    batch_size = 10
    data_iter = load_data((features, labels), batch_size, is_train=True)

    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss()

    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    num_epochs = 3
    for epoch in range(num_epochs):
        for x, y in data_iter:
            l = loss(net(x), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f"epoch {epoch}, loss {l.item()}")

    print("估算的 w:", net[0].weight.data)
    print("估算的 b:", net[0].bias.data)

if __name__ == "__main__":
    main()