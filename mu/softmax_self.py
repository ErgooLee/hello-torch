import torch
from Accumulator import Accumulator
from softmax_data import load_data_fashion_mnist
import my_lib
from Animator import Animator
from torch import nn
import matplotlib.pyplot as plt


def softmax(x):
    x_exp = torch.exp(x)
    partition = x_exp.sum(dim=1, keepdim=True)
    return x_exp / partition

def net(x, w, b):
    reshaped_x = x.reshape(-1, w.shape[0])
    return torch.mm(reshaped_x, w) + b

def cross_entropy(y_hat, y):
    y_prob = softmax(y_hat)
    return -torch.log(y_prob[range(y_prob.shape[0]), y])

def correct_num(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


# 5. 评估模型在数据集上的准确率
def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    for x, y in data_iter:
        metric.add(correct_num(net(x), y), y.numel())
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):
    metric = Accumulator(3)
    for x, y in train_iter:
        y_hat = net(x)
        l = loss(y_hat, y).mean()
        l.backward()
        updater()
        metric.add(float(l.detach() * y.numel()), correct_num(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


# 训练一个迭代周期（Epoch）- 官方库版本
def train_epoch_ch3_lib(net, train_iter, loss, updater):
    net.train()
    metric = Accumulator(3)
    for x, y in train_iter:
        y_hat = net(x)
        l = loss(y_hat, y).mean()
        updater.zero_grad()
        l.backward()
        updater.step()
        metric.add(float(l.detach() * y.numel()), correct_num(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


# 7. 完整的训练函数
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater, train_method):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.0, 1.5],
                        legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        train_loss, train_acc = train_method(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, (train_loss, train_acc, test_acc))
        print(f"Epoch {epoch + 1}: Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")


# 1. 将数字类别标签转换为文本标签
def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 2. 预测并在窗口中展示结果
def predict_ch3(net, test_iter, n=6):
    """预测图像标签并展示结果"""
    for X, y in test_iter:
        break

    # 此时网络输出的是 Logits，利用 argmax 依然能正确获取最大概率对应的索引
    y_hat = net(X)
    preds = y_hat.argmax(axis=1)

    true_labels = get_fashion_mnist_labels(y[:n])
    pred_labels = get_fashion_mnist_labels(preds[:n])
    titles = [f"True: {t}\nPred: {p}" for t, p in zip(true_labels, pred_labels)]

    imgs = X[:n].reshape(n, 28, 28)

    fig, axes = plt.subplots(1, n, figsize=(n * 2.2, 3))
    for i in range(n):
        ax = axes[i]
        img = imgs[i].detach()
        if torch.is_tensor(img):
            img = img.numpy()

        ax.imshow(img, cmap='gray')
        ax.set_title(titles[i], fontsize=9, color='green' if true_labels[i] == pred_labels[i] else 'red')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


# 从零开始实现的完整训练流程
def train_self():
    # 数据加载
    batch_size = 64
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    num_inputs = 28 * 28
    num_outputs = 10

    # 初始化权重和偏置
    w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    # 定义学习率与 Updater
    lr = 0.1

    net_fn = lambda x: net(x, w, b)
    updater_fn = lambda: my_lib.sgd([w, b], lr)

    # 开始训练
    num_epochs = 10
    plt.ion()
    train_ch3(net_fn, train_iter, test_iter, cross_entropy, num_epochs, updater_fn, train_epoch_ch3)
    plt.ioff()
    plt.show()
    predict_ch3(net_fn, test_iter)



# 官方库实现的完整训练流程
def train_nn():
    # 数据加载
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    # 定义学习率与 Updater
    lr = 0.1

    nn_linear = nn.Linear(28 * 28, 10)
    net = nn.Sequential(nn.Flatten(), nn_linear)

    # 官方交叉熵函数内置了 Softmax 变换
    loss = nn.CrossEntropyLoss()

    updater = torch.optim.SGD(net.parameters(), lr=lr)

    # 开始训练
    num_epochs = 10
    plt.ion()
    train_ch3(net, train_iter, test_iter, loss, num_epochs, updater, train_epoch_ch3_lib)
    plt.ioff()
    plt.show()
    predict_ch3(net, test_iter)


if __name__ == "__main__":
    train_self()  # 您也可以随时替换为 train_self() 进行对比测试