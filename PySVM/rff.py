import numpy as np
from sklearn.base import TransformerMixin

class NormalRFF(TransformerMixin): #随机傅里叶特征逼近RBF核函数
    def __init__(self, gamma = 1, D = 1000) -> None:
        super().__init__()
        self.gamma = gamma
        self.D = D

    def fit(self, X: np.ndarray):
        self.n_features = np.array(X).shape[1] #特征数
        self.w = np.sqrt(self.gamma * 2) * np.random.randn(
            self.D, self.n_features) #np.random.randn(D, n_features) 生成的矩阵中，每一个位置上的数字都是独立地从标准正态分布 N(0, 1) 中随机抽取出来的，有 D 行，n_features 列
                                     #np.random.randn(self.D, self.n_features)：产生一个形状为 (D, d) 的矩阵，其中的每个元素都服从 标准正态分布 N(0, 1)
                                     #np.sqrt(self.gamma * 2)：这是关键的比例因子。在概率论中，如果你有一个变量 z ~ N(0, 1)，那么变量 az 的分布就是 N(0, a^2)。我们想要的方差是 2 * gamma。所以，我们需要乘以标准差 sqrt{2 * gamma}
                                     #要生成的 w 满足 w ~ N(0, 2 * gamma * I)，直接采样一个协方差为 2 * gamma * I 的多元正态分布，等价于采样一个标准正态分布后再乘以标准差 sqrt{2 * gamma}，因为 np.random.randn 生成标准正态分布的速度非常快
        self.b = np.random.uniform(0, 2 * np.pi, self.D) #b 服从 [0, 2 * pi] 的均匀分布
        return self

    def transform(self, X: np.ndarray):
        return np.sqrt(2 / self.D) * np.cos(np.matmul(X, self.w.T) + self.b)

    def fit_transform(self, X: np.ndarray):
        return self.fit(X).transform(X)