import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 解决 Mac 系统中文字体缺失及负号显示问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False


def run_experiment(parameters, numbers, decay_value, noise_rate):
    # 设置随机种子，保证每次运行结果完全一致
    torch.manual_seed(42)
    np.random.seed(42)

    # ================= 1. 构造数据集 =================
    true_w = torch.zeros(parameters)
    true_w[0] = 3.0  # 前2个是有用特征
    true_w[1] = -2.0  # 其余全是垃圾特征

    # (1) 训练集
    X_train = torch.randn(numbers, parameters)
    train_noise = torch.randn(numbers) * noise_rate
    y_train = (X_train @ true_w + train_noise).unsqueeze(1)

    # (2) 测试集：1000 个全新的未知样本
    n_test = 1000
    X_test = torch.randn(n_test, parameters)
    test_noise = torch.randn(n_test) * noise_rate
    y_test = (X_test @ true_w + test_noise).unsqueeze(1)

    # ================= 2. 定义训练函数 =================
    def train(weight_decay_val):
        model = nn.Linear(parameters, 1, bias=False)
        nn.init.normal_(model.weight, 0, 1.0)  # 初始权重设为 1.0

        criterion = nn.MSELoss()
        optimizer = optim.SGD(
            model.parameters(), lr=0.05, weight_decay=weight_decay_val
        )

        for epoch in range(500):
            optimizer.zero_grad()
            pred = model(X_train)
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()

        return model.weight.detach().numpy().flatten()

    # ================= 3. 运行对比训练 =================
    w_no_decay = train(weight_decay_val=0.0)  # 模型 A：无衰退
    w_with_decay = train(weight_decay_val=decay_value)  # 模型 B：有衰退

    # ================= 4. 在测试集上评估 Loss =================
    criterion = nn.MSELoss()

    pred_no_decay = X_test @ torch.tensor(
        w_no_decay, dtype=torch.float32
    ).unsqueeze(1)
    test_loss_no_decay = criterion(pred_no_decay, y_test).item()

    pred_with_decay = X_test @ torch.tensor(
        w_with_decay, dtype=torch.float32
    ).unsqueeze(1)
    test_loss_with_decay = criterion(pred_with_decay, y_test).item()

    # ================= 5. 控制台打印详细结果 =================
    print("\n" + "=" * 70)
    print(
        f"【实验配置】特征数: {parameters} | 训练样本数: {numbers} | 权重衰退: {decay_value}"
    )
    print("=" * 70)
    print(
        f"{'特征名称':<12} | {'真实权重':<8} | {'无衰退学到的 w':<16} | {'有衰退学到的 w':<16}"
    )
    print("-" * 70)

    # 动态生成特征名，最多打印前 10 个特征（防止 100 个刷屏）
    display_limit = min(parameters, 10)
    for i in range(display_limit):
        fname = (
            f"x{i + 1} (有用)"
            if i < 2
            else (f"x{i + 1} (垃圾)" if parameters <= 10 else f"x{i + 1}")
        )
        print(
            f"{fname:<12} | {true_w[i].item():<8.2f} | {w_no_decay[i]:<16.4f} | {w_with_decay[i]:<16.4f}"
        )

    if parameters > 10:
        print(f"... 剩余 {parameters - 10} 个垃圾特征已省略 ...")
    print("=" * 70)

    improvement = (
                          (test_loss_no_decay - test_loss_with_decay) / test_loss_no_decay
                  ) * 100
    print(f"❌ 无权重衰退测试集 Loss : {test_loss_no_decay:.4f}")
    print(f"✅ 有权重衰退测试集 Loss : {test_loss_with_decay:.4f}")
    print(f"👉 权重衰退带来的测试集表现提升: {improvement:.2f}%\n")

    # ================= 6. Matplotlib 可视化图表 =================
    plt.figure(figsize=(14, 5))

    # 子图 1：只画前 10 个特征的权重对比，保证图表清晰
    plt.subplot(1, 2, 1)
    plot_num = min(parameters, 10)
    x_indices = np.arange(plot_num)
    bar_width = 0.25

    plt.bar(
        x_indices - bar_width,
        true_w[:plot_num].numpy(),
        width=bar_width,
        label="True Weights",
        color="gray",
        alpha=0.5,
    )
    plt.bar(
        x_indices,
        w_no_decay[:plot_num],
        width=bar_width,
        label="No Decay",
        color="salmon",
    )
    plt.bar(
        x_indices + bar_width,
        w_with_decay[:plot_num],
        width=bar_width,
        label=f"With Decay ({decay_value})",
        color="cornflowerblue",
    )

    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plot_labels = [f"x{i + 1}" for i in range(plot_num)]
    plot_labels[0] += "\n(有用)"
    plot_labels[1] += "\n(有用)"
    plt.xticks(x_indices, plot_labels)
    plt.ylabel("Weight Value")
    plt.title(f"Learned Weights (Showing top {plot_num}/{parameters})")
    plt.legend()
    plt.grid(axis="y", linestyle=":", alpha=0.5)

    # 子图 2：测试集 Loss 对比柱状图
    plt.subplot(1, 2, 2)
    models = ["No Decay", f"With Decay ({decay_value})"]
    losses = [test_loss_no_decay, test_loss_with_decay]
    colors = ["salmon", "cornflowerblue"]

    bars = plt.bar(models, losses, color=colors, width=0.4)
    plt.ylabel("Test MSE Loss (Lower is Better)")
    plt.title(f"Test Loss (P={parameters}, N={numbers})")
    plt.grid(axis="y", linestyle=":", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 实验 1: N > P (20 > 5), 惩罚过重 -> 欠拟合 (无衰退胜)
    run_experiment(parameters=5, numbers=20, decay_value=0.1, noise_rate=0.1)

    # 实验 2: N > P (20 > 5), 调小惩罚 -> 适度平衡
    run_experiment(parameters=5, numbers=20, decay_value=0.01, noise_rate=0.1)

    run_experiment(parameters=5, numbers=20, decay_value=0.01, noise_rate=0.5)

    # 实验 3: P >> N (100 >> 20), 严重过拟合 -> 权重衰退大显神威！(有衰退暴击无衰退)
    run_experiment(parameters=100, numbers=20, decay_value=0.1, noise_rate=0.1)
