import torch


def trop_out_layer(x, drop_rate):
    assert 0 <= drop_rate <= 1
    if drop_rate == 1:
        return torch.zeros_like(x)
    if drop_rate == 0:
        return x
    # 按概率存活
    mask = (torch.rand(x.shape) > drop_rate).float()
    # mask * x 是点乘，不是x乘
    # x = p * x + (1 - p) * x / (1-p)
    return mask * x / (1.0 - drop_rate)


if __name__ == '__main__':
    x = torch.arange(0, 16).reshape(2, 8)
    print(trop_out_layer(x, 0))
    print(trop_out_layer(x, 1))
    print(trop_out_layer(x, 0.5))
