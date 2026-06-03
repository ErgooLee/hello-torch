import matplotlib.pyplot as plt
from IPython import display


class Animator:
    """在动画中绘制数据的实用工具类（适用于 Jupyter Notebook）"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(6, 4)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend
        self.fmts = fmts
        self.xlim = xlim
        self.ylim = ylim
        self.xscale = xscale
        self.yscale = yscale
        self.X, self.Y = None, None

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if self.X is None:
            self.X = [[] for _ in range(n)]
        if self.Y is None:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.ax.cla()
        for i, (x_arr, y_arr) in enumerate(zip(self.X, self.Y)):
            self.ax.plot(x_arr, y_arr, self.fmts[i], label=self.legend[i])
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        if self.xlim:
            self.ax.set_xlim(self.xlim)
        if self.ylim:
            self.ax.set_ylim(self.ylim)
        self.ax.set_xscale(self.xscale)
        self.ax.set_yscale(self.yscale)
        if self.legend:
            self.ax.legend()

        # 核心：清除当前输出并重新显示图像以实现“动画”效果
        # display.display(self.fig)
        # display.clear_output(wait=True)
        plt.draw()
        plt.pause(0.1)