class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        # 初始化一个长度为 n 的列表，初始值全部为 0.0
        self.data = [0.0] * n

    def add(self, *args):
        # 将传入的参数依次加到 self.data 对应的位置上
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        # 重置所有累加器为 0.0
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        # 允许通过索引（如 metric[0]）来访问累加的数据
        return self.data[idx]