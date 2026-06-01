import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch import nn


def synthetic_data(w, b, num_examples):
    """y = Xw +b + c"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.mm(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)


def load_data(data_array, batch_size, is_train):
    data_set = data.TensorDataset(*data_array)
    return DataLoader(data_set, batch_size=batch_size, shuffle=is_train)


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