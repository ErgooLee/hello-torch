import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt


# 2. 标签转换函数
def get_fashion_mnist_labels(labels):
    text_labels = [
        "t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt",
        "sneaker", "bag", "ankle boot"
    ]
    # 支持传入张量或列表
    return [text_labels[int(i)] for i in labels]


# 3. 补全后的 show_images 函数
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.squeeze().numpy(), cmap='gray')
        else:
            ax.imshow(img, cmap='gray')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i], fontsize=9)
    return axes


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST("../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST("../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0))

if __name__ == "__main__":
    train_data, test_data = load_data_fashion_mnist(18)
    X, y = next(iter(train_data))

    show_images(X, num_rows=2, num_cols=9, titles=get_fashion_mnist_labels(y))
    plt.show()  # 弹出窗口展示图片
