import torch
from matplotlib import pyplot as plt

from mu import my_lib
from softmax_self import load_data_fashion_mnist, train_ch3, predict_ch3, train_epoch_ch3, \
    cross_entropy


def relu(x):
    a = torch.zeros_like(x)
    return torch.max(x, a)


def net(x, w1, b1, w2, b2):
    x = x.reshape((-1, w1.shape[0]))
    h = relu(x @ w1 + b1)
    o = h @ w2 + b2
    return o

def main():
    # 数据加载
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    num_inputs = 28 * 28
    num_hidden = 256
    num_outputs = 10

    # 初始化权重和偏置
    w1 = torch.normal(0, 0.01, size=(num_inputs, num_hidden), requires_grad=True)
    b1 = torch.zeros(num_hidden, requires_grad=True)
    w2 = torch.normal(0, 0.01, size=(num_hidden, num_outputs), requires_grad=True)
    b2 = torch.zeros(num_outputs, requires_grad=True)

    # 定义学习率与 Updater
    lr = 0.1

    net_fn = lambda x: net(x, w1, b1, w2, b2)
    updater_fn = lambda: my_lib.sgd([w1, b1, w2, b2], lr)

    # 开始训练
    num_epochs = 10
    plt.ion()
    train_ch3(net_fn, train_iter, test_iter, cross_entropy, num_epochs, updater_fn, train_epoch_ch3)
    plt.ioff()
    plt.show()
    predict_ch3(net_fn, test_iter)


if __name__ == "__main__":
    main()
