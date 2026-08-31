import torch
import torch.nn as nn


def show_conv_opera():
    # 5×5 输入
    x = torch.arange(1, 26, dtype=torch.float32).reshape(5, 5)

    # PyTorch Conv2d 要求输入格式：
    # [batch, channel, height, width]
    x = x.reshape(1, 1, 5, 5)

    # 创建一个 3×3 卷积
    conv = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        bias=False
    )

    # 手动指定卷积核
    conv.weight.data = torch.tensor([[
        [
            [1., 0., -1.],
            [1., 0., -1.],
            [1., 0., -1.]
        ]
    ]])

    # 执行卷积
    y = conv(x)

    print(y)


def show_conv_opera_multi_channel():

    # 一个 RGB 图片：3 个通道，每个 5×5
    x = torch.tensor([
        # R
        [
            [1., 2., 3., 4., 5.],
            [6., 7., 8., 9., 10.],
            [11., 12., 13., 14., 15.],
            [16., 17., 18., 19., 20.],
            [21., 22., 23., 24., 25.]
        ],

        # G
        [
            [2., 2., 2., 2., 2.],
            [2., 2., 2., 2., 2.],
            [2., 2., 2., 2., 2.],
            [2., 2., 2., 2., 2.],
            [2., 2., 2., 2., 2.]
        ],

        # B
        [
            [3., 3., 3., 3., 3.],
            [3., 3., 3., 3., 3.],
            [3., 3., 3., 3., 3.],
            [3., 3., 3., 3., 3.],
            [3., 3., 3., 3., 3.]
        ]
    ])

    # [C, H, W] → [N, C, H, W]
    x = x.unsqueeze(0)

    conv = nn.Conv2d(
        in_channels=3,
        out_channels=1,
        kernel_size=3,
        bias=False
    )

    # 设置一个 3×3×3 的卷积核
    conv.weight.data = torch.tensor([[
        # R 通道的卷积核
        [
            [1., 0., -1.],
            [1., 0., -1.],
            [1., 0., -1.]
        ],

        # G 通道的卷积核
        [
            [1., 0., -1.],
            [1., 0., -1.],
            [1., 0., -1.]
        ],

        # B 通道的卷积核
        [
            [1., 0., -1.],
            [1., 0., -1.],
            [1., 0., -1.]
        ]
    ]])

    y = conv(x)

    print(y.shape)
    print(y)


if __name__ == '__main__':
    show_conv_opera()
    show_conv_opera_multi_channel()
