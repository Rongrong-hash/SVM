import numpy as np
from sklearn.base import TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel


class NystromRBF(TransformerMixin):
    def __init__(self, gamma=1.0, n_components=200, random_state=None, jitter=1e-6):
        super().__init__()
        self.gamma = gamma
        self.n_components = n_components #从原始数据集中随机挑选出来的代表性样本（地表点）数量
        self.random_state = random_state #随机数种子，随机挑选样本作为地标。设置固定种子可以保证每次运行代码时，选中的点都是一样的，方便结果复现
        self.jitter = jitter #数值稳定性微调项（抖动项），防止程序在进行矩阵运算时因为“除以 0”或者“计算不可逆矩阵”而崩溃

    def _get_rng(self): #随机数生成器（Random Number Generator）的统一管理器。在 Python 数据科学中，处理随机性有一个痛点：有时候希望结果完全随机，有时候为了调试，希望每次运行的结果都一模一样。这个函数就是为了解决这个问题
        if isinstance(self.random_state, np.random.Generator): #检查用户传入的 random_state 是不是已经是一个现成的“生成器对象”。如果是，就直接拿来用。
            return self.random_state
        return np.random.default_rng(self.random_state) #NumPy 官方推荐的新一代随机数创建方式：
                                                        #如果传入一个整数，创建一个固定序列的生成器（每次运行选出的地标点都一样）。
                                                        #如果传入 None，创建一个真正随机的生成器（每次运行选出的地标点都不同）。

    def fit(self, X: np.ndarray): #从原始数据中提取出能够代表全局特征的“骨架”
        X = np.asarray(X)
        n_samples = X.shape[0] #样本数量
        m = min(self.n_components, n_samples) #需要从数据集中抽取 m 个样本：
                                              #情况 A：正常情况（数据量大）：用户设置 n_components = 200，而你有 10,000 个样本，算法正常从 10,000 个点里选出 200 个代表点
                                              #情况 B：极端情况（数据量过小）：用户设置 n_components = 200，但你实际只有 50 个样本，无法变出 200 个不重复的代表点，所以代码“退而求其次”，把这 50 个样本全部当作地标点

        rng = self._get_rng() #返回一个 NumPy 随机数生成器对象 (Generator)
        indices = rng.choice(n_samples, size=m, replace=False) #选出那 m 个具有代表性的地标点索引
        self.landmarks_ = X[indices] #选出的 m 个地表点样本

        K_mm = rbf_kernel(self.landmarks_, self.landmarks_, gamma=self.gamma) #计算所有地标点（Landmarks）两两之间的相似度
                                                                              #矩阵的第 i 行、第 j 列：存储的是第 i 个地标点和第 j 个地标点之间的 RBF（高斯）核函数值
        K_mm += self.jitter * np.eye(m) #岭回归正则化（Tikhonov Regularization） 或 添加抖动项（Adding Jitter）
                                        #np.eye(m)：创建一个 m * m 的单位矩阵（对角线全是 1，其他地方全是 0）
                                        #self.jitter * ...：把这个单位矩阵乘以一个极小的数（比如 10^{-6}）
                                        #+=：把这个微小的对角阵加到你刚算好的 K_mm 上
                                        #结果：K_mm 的对角线元素从原来的 $1.0$ 变成了 $1.000001$，而其他位置保持不变

        eigvals, eigvecs = np.linalg.eigh(K_mm) #eigvals (特征值 Lambda)：一个向量，代表了每个特征方向的重要程度（能量/方差）
                                                #eigvecs (特征向量 U)：一个矩阵，每一列代表了一个主成分方向
                                                #对称矩阵可以分解为：K_{mm} = U * Lambda * U^T
        keep = eigvals > 1e-12 #识别并保留那些真正有意义的特征方向，丢弃掉那些由于数值计算误差或信息冗余产生的“垃圾”信号，在理想的数学世界里，特征值应该要么很大，要么正好是 0。但在计算机的浮点数运算中，由于精度有限，原本应该是 0 的地方往往会变成一个极其微小的数字
        self.components_ = eigvecs[:, keep] / np.sqrt(eigvals[keep]) #切片（Filtering）：eigvecs[:, keep] 只选出了那些通过了刚才 10^{-12} 门槛的特征向量
                                                                     #缩放（Scaling）：将特征向量除以特征值的平方根 np.sqrt(eigvals[keep])
                                                                     #存储（Saving）：把这个最终的换算矩阵存入 self.components_
                                                                     #计算 U * Lambda^{-1/2}
                                                                     #要近似核矩阵 K 约等于 K_{nm} * K_{mm}^{-1} * K_{mn}。为了得到显式映射 phi(X)，需要把中间的 K_{mm}^{-1} 拆成两半：K_{mm}^{-1} = (U * Lambda^{-1/2}) * (U * Lambda^{-1/2})^T
        return self

    def transform(self, X: np.ndarray):
        if not hasattr(self, "landmarks_"): #检查当前这个类实例（self）中是否存在名为 landmarks_ 的属性
            raise RuntimeError("fit must be called before transform.")

        X = np.asarray(X)
        K_nm = rbf_kernel(X, self.landmarks_, gamma=self.gamma) #测量每一个“新样本”与选出的“代表（地标点）”之间的相似程度
                                                                #X (输入数据): 维度为 (N_{samples}, d)
                                                                #self.landmarks_ (地标点): 维度为 (M_{landmarks}, d)
                                                                #K_nm (结果矩阵): 维度为 (N_{samples}, M_{landmarks})
        return K_nm @ self.components_ #phi(X) 约等于 K_{nm} * (K_{mm}^{-1/2})

    def fit_transform(self, X: np.ndarray):
        return self.fit(X).transform(X)
