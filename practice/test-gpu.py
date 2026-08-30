import torch
import torch.nn as nn



def get_device():
    # 自动判断并选择最佳设备: CUDA > MPS > CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


if __name__ == '__main__':
    device = get_device()
    print(f"当前使用的设备是: {device}")

    # 1. 创建 Tensor 到目标设备
    data = torch.randn(32, 64, device=device)

    # 2. 将神经网络模型移动到目标设备
    model = nn.Linear(64, 10).to(device)

    # 3. 在 GPU 上执行前向传播
    output = model(data)
    print(output.device)  # 输出: mps:0
