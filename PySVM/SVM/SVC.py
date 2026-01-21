import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from ..rff import NormalRFF
from ..Nyström import NystromRBF
from ..solver import Solver, SolverWithCache

class BiLinearSVC(BaseEstimator): #BaseEstimator是sklearn中的基类，用于实现自定义的分类器，使自定义模型能够无缝接入 sklearn 的生态系统
    def __init__(self,
                 C: float = 1.,  #惩罚参数
                 max_iter: int = 1000, #最大迭代次数，防止SMO不收敛而陷入死循环
                 tol: float = 1e-5, #容忍度，用于控制迭代停止的精度，在实际计算中，由于浮点数精度和计算效率的限制，我们几乎不可能让所有样本完全精确地满足KKT条件，不是松弛变量：
                                    #样本状态           |理论KKT条件       |代码中的违反判定（与 tol 比较）
                                    #间隔外 (αi​=0)      |yi​Ei​⩾0           |yi​Ei​<−tol                    
                                    #间隔上 (0<αi​<C)    |yi​Ei​=0           |tol<yi​Ei or yi​Ei<-tol
                                    #间隔内/误分 (αi​=C)  |yi​Ei​⩽0          |yi​Ei​>tol 
                cache_size: int = 256, #SMO 算法中，每一次迭代都需要挑选两个 alpha 进行坐标上升优化，需要频繁用到 Q 矩阵，Q_{ij} = y_i y_j ({x}_i^t {x}_j)，如果使用缓存，算法会将计算过的点积结果存入内存，下次需要用到相同的i和j时，直接从内存读取
                shrinking: bool = False,
                ) -> None:
        super().__init__() #执行父类 BaseEstimator 的初始化逻辑
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.cache_size = cache_size
        self.shrinking = shrinking

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = np.array(X), np.array(y, dtype=float) #创建参数X, y的类型为ndarray的副本
        y[y != 1] = -1 #如果 y = [0, 1, 0, 2] 那么 y != 1 会得到 [True, False, True, True]y[...]；括号里放入这个布尔数组，NumPy 就会只选中那些对应位置为 True 的元素；= -1 将选中的所有位置的值一次性改为 -1
        l, self.n_features = X.shape #l 是 X 的行数（样本数量），self.n_features 是 X 的列数（特征维度）
        p = -np.ones(l) #SVM 的对偶问题目标函数通常写作：min 1/2 * {alpha}^t * Q * {alpha} - sum(alpha_i), 这里的 sum(alpha_i) 可以改写为向量内积的形式：{e}^t * {alpha}，其中 {e} 是全 1 向量。在通用的二次规划（QP）求解器中，标准的目标函数形式通常是：min 1/2 * {x}^t * Q * {x} + {p}^t * {x}，故令 p = [-1，-1，..., -1]^t

        w = np.zeros(self.n_features) #权重 {w}
        if self.cache_size == 0:
            Q = y.reshape(-1, 1) * y * (X @ X.T) #np.matmul(X, X.T)：计算特征矩阵 X 与其转置的乘积。结果是一个 l*l 的矩阵，其中第 (i, j) 个元素就是样本 x_i 和 x_j 的点积
                                                         #y.reshape(-1, 1) * y：把列向量 y 乘以行向量 y，生成一个 l*l 的符号矩阵。如果 y_i 和 y_j 同号，结果为 1，异号则为 -1，当两个数组的形状不匹配时，NumPy 会尝试自动扩展较小的数组以匹配较大数组的形状，从而进行逐元素（Element-wise）运算
                                                         #相乘结果：得到的 Q 矩阵存储了对偶问题所需的所有系数
                                                         #注意这里的 * 是广播机制，不是矩阵乘法
            solver = Solver(Q, p, y, self.C, self.tol, self.shrinking)
        else:
            solver = SolverWithCache(p, y, self.C, self.tol, self.cache_size, self.shrinking) #按需计算：只有当 SMO 算法需要用到某个 Q_{ij} 时，才会去计算那两个向量的点积
                                                                              #LRU 缓存：计算后的结果会存入缓存（大小由 cache_size 决定）。如果内存满了，就丢弃旧的，存入新的
        def func(i): #计算 Q 矩阵的第 i 列。在 SMO 算法中，当我们决定更新第 i 个拉格朗日乘子 alpha_i 时，我们需要知道该样本与所有其他样本之间的关系。这个函数就是在“按需”计算这些关系
            return y * (X @ X[i]) * y[i] #np.matmul(X, X[i]): 这是计算特征矩阵 X（所有样本）与第 i 个样本 X[i] 的点积。结果是一个长度为 l 的向量，其中包含了 (x_0^t * x_i, x_1^t * x_i,…, x_{l-1}^t * x_i)
                                                    #* y[i]:将上述所有点积结果乘以第 i 个样本的标签
                                                    #y * ...:再乘以所有样本的标签向量 y。由于这里是元素对元素的乘法，它实际上完成了 y_j * y_i 的操作
        
        for n_iter in range(self.max_iter): #SMO 算法的主循环
            i, j = solver.select_and_run() #选择工作集，求解器会扫描所有样本，寻找最违反 KKT 条件（即最不符合分类规则）的两个样本索引 i 和 j
            if i < 0: #如果返回的 i < 0，说明所有样本都已经满足了 tol 设定的精度要求，或者已经没有优化的空间了，算法提前收敛，跳出循环
                break

            delta_i, delta_j = solver.update(i, j, func) #解析更新，利用之前定义的 func（计算 Q 矩阵列的函数）来获取必要的点积值，delta_i 和 delta_j 分别代表 alpha_i 和 alpha_j 的变化量（新值减去旧值）
            w += delta_i * y[i] * X[i] + delta_j * y[j] * X[j] #由于 w = sum(alpha_k * y_k * x_k)，当 alpha_i 改变了 Delta(alpha_i) 时，w 的变化量就是 Delta(alpha_i * y_i * x_i)
        else:
            print("LinearSVC not converge with {} iterations".format(self.max_iter)) #如果循环是通过 break 正常退出（即模型收敛了），else 块不会执行；如果循环跑满了 max_iter 次还没有触碰到 break，则说明模型在规定步数内没有收敛，此时会触发 else 打印警告信息
        
        self.coef_ = (w, solver.calculate_rho()) #将训练好的模型参数正式“封存”到类变量中，以便后续进行预测
                                                    #{w} 是权重向量，solver.calculate_rho() 是截距，虽然 {w} 可以随 alpha 实时更新，但截距 rho 通常在迭代结束后，利用支持向量来确定
                                                    #命名为 self.coef_ 是为了遵循 Scikit-learn 的命名规范，在 sklearn 生态中，所有在 fit 过程中学到的参数都必须以双下划线结尾
        return self #在 Scikit-learn 的设计模式中，return self 是一个非常关键的约定俗成写法。意思是在 fit 方法执行完毕后，返回已经训练好的模型实例本身

    def decision_function(self, X: np.ndarray) -> np.ndarray: #决策函数，输出预测值
        return self.coef_[0] @ np.array(X).T - self.coef_[-1] #self.coef_[0]:在 fit 阶段存入的权重向量 {w}，维度是 (n_features,)
                                                                        #np.array(X).T，维度是(n_features, m_samples)
                                                                        #np.matmul(...):计算权重 {w} 与所有样本特征的点积。结果是一个长度为 m_samples 的向量
                                                                        #self.coef_[-1]:在 fit 阶段存入的截距 rho（偏置项 b），-是广播机制
    
    def predict(self, X: np.ndarray) -> np.ndarray: #预测函数，输出预测标签(0-1)
        return (self.decision_function(np.array(X)) >= 0).astype(int) #... >= 0 得到的结果是一个布尔数组，.astype(int) 将布尔值转换为整数
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float: #评估函数，给定特征和标签，输出正确率
        y_standardized = (np.array(y) == 1).astype(int) #确保传入的 y 被转换成与 predict 输出一致的 0-1 格式，而不是-1-1格式
        return accuracy_score(y_standardized, self.predict(X))


class LinearSVC(BiLinearSVC): #多分类线性SVM，使用sklearn的multiclass模块实现了多分类
    def __init__(self,
                 C: float = 1.,
                 max_iter: int = 1000,
                 tol: float = 1e-5,
                 cache_size: int = 256,
                 shrinking: bool = False,
                 multiclass: str = "ovr", #One-vs-Rest 分类策略
                 n_jobs=None) -> None: #指定训练时使用的 CPU 核心数量
        super().__init__(C, max_iter, tol, cache_size, shrinking)
        self.multiclass = multiclass
        self.n_jobs = n_jobs
        params = {
            "estimator": BiLinearSVC(C, max_iter, tol, cache_size, shrinking),
            "n_jobs": n_jobs
        }
        self.multiclass_model: OneVsOneClassifier = {
            "ovo": OneVsOneClassifier(**params),
            "ovr": OneVsRestClassifier(**params),
        }[multiclass] #{ "ovo": ..., "ovr": ... }：定义了一个临时字典，里面存了两个已经实例化好的“战术模型”
                      #[multiclass]：根据变量 multiclass 的值（即 "ovo" 或 "ovr"）作为键（Key），从字典里把对应的模型“拿”出来
                      #最后把拿出来的模型存入 self.multiclass_model 中
                      #**params 字典拆解，把字典里的东西拆开，直接传给分类器
                      
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.multiclass_model.fit(X, y)
        return self

    def decision_function(self, X: np.ndarray):
        return self.multiclass_model.decision_function(X)

    def predict(self, X: np.ndarray):
        return self.multiclass_model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray):
        return self.multiclass_model.score(X, y)


class BiKernelSVC(BiLinearSVC): #二分类核SVM，该类被多分类KernelSVC继承，所以不需要使用它。优化问题与BiLinearSVC相同，只是Q矩阵定义不同
    def __init__(self,
                 C: float = 1.,
                 kernel: str = 'rbf', #核函数有:'linear'、'poly'、'rbf'、'sigmoid'，默认径向基函数(RBF)，计算两个样本在高维空间中的“相似度”，而无需真正地进行高维映射计算
                 degree: float = 3, #仅在 kernel='poly' 时有效，决定了多项式的最高次数
                 gamma: str = 'scale', #适用范围是：'rbf', 'poly', 'sigmoid' 核，控制了单个训练样本影响范围的大小
                 coef0: float = 0., #适用范围是：'poly', 'sigmoid', 核函数公式中的常数,调节模型受高阶项影响的程度
                 max_iter: int = 1000,
                 rff: bool = False, #传统的核 SVM 计算复杂度随样本量 N 呈平方级增长（需要计算 N * N 的核矩阵）。RFF (Random Fourier Features) 通过随机采样的方法，将 RBF 核近似映射为显式的线性特征
                 nystrom: bool = False, #Nyström 采样
                 D: int = 1000, #仅在 rff=True 时有效,指将原始特征映射到多少维的随机空间，D 越大，对 RBF 核的近似越精确
                 tol: float = 1e-5,
                 cache_size: int = 256,
                 shrinking: bool = False) -> None:
        super().__init__(C, max_iter, tol, cache_size, shrinking)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.rff = rff
        self.nystrom = nystrom
        self.D = D
        self.transformer = None #用于存储 RFF 或 Nystrom 映射器

    def register_kernel(self, X, current_kernel): #注册核函数
        if type(self.gamma) == str:
            gamma = 1 / (X.shape[1] * X.std()) if self.gamma == 'scale' else 1 / X.shape[1]
        else:
            gamma = self.gamma #如果直接传入一个数字（如 0.1），它就直接使用该数字

        degree = self.degree
        coef0 = self.coef0
        return {
            "linear": lambda x, y: x @ y.T,
            "poly": lambda x, y: (gamma * (x @ y.T) + coef0)**degree,
            "rbf": lambda x, y: np.exp(-gamma * ((x**2).sum(1, keepdims=True) + (y**2).sum(1) - 2 * (x @ y.T))),
            "sigmoid": lambda x, y: np.tanh(gamma * (x @ y.T) + coef0)
        }[current_kernel]

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = np.array(X), np.array(y, dtype=float)
        y[y != 1] = -1
        l, self.n_features = X.shape

        current_kernel = self.kernel 

        if self.rff or self.nystrom:
            #计算gamma
            if type(self.gamma) == str:
                gamma = {'scale': 1 / (self.n_features * X.std()), 
                         'auto': 1 / self.n_features}[self.gamma]
            else:
                gamma = self.gamma

            if self.rff:
                #初始化并保存映射器，以便 predict 时使用
                self.transformer = NormalRFF(gamma, self.D).fit(X)
            elif self.nystrom:
                self.transformer = NystromRBF(gamma=gamma, n_components=self.D).fit(X)
            
            #只在 fit 开始时转换一次 X, 转换后的 X 形状为 (l, D)
            X = self.transformer.transform(X)
            #之后的操作全部视为线性核
            current_kernel = "linear"

        #注册核函数逻辑（如果是 RFF，此时 kernel 已经是 linear 了）
        kernel_func = self.register_kernel(X, current_kernel)

        p = -np.ones(l)

        if self.cache_size == 0:
            Q = y.reshape(-1, 1) * y * kernel_func(X, X)
            solver = Solver(Q, p, y, self.C, self.tol, self.shrinking)
        else:
            solver = SolverWithCache(p, y, self.C, self.tol, self.cache_size, self.shrinking)

        def func(i):
            #此时 X 已经是映射后的 Z 矩阵，kernel_func 是 np.matmul(x, y.T), 复杂度从 O(N*D) 降到了 O(N)
            return y * kernel_func(X, X[i:i+1]).flatten() * y[i] #X[i] 返回的是一个形状为 (d,) 的 向量。X[i:i+1] 返回的是一个形状为 (1, d) 的 矩阵（切片保留维度）。而 kernel_func（无论是 RFF 还是标准 RBF）通常期望输入是矩阵以便进行批量计算，所以用 X[i:i+1] 可以避免维度报错，确保输出是一个 (N, 1) 的矩阵

        for n_iter in range(self.max_iter):
            i, j = solver.select_and_run()
            if i < 0:
                break
            solver.update(i, j, func)
        else:
            print("KernelSVC not converge with {} iterations".format(
                self.max_iter))

        if self.rff or self.nystrom:
            self.decision_function = lambda x: (solver.alpha * y) @ (
                X @ self.transformer.transform(x).T) - solver.calculate_rho() #solver.alpha * y: 对应 alpha_i * y_i
                                                                              #kernel_func(X, x): 对应 K(x_i, x)。计算训练集 X 中所有样本与新样本 x 的核相似度
        else:
            self.decision_function = lambda x: (solver.alpha * y) @ kernel_func(X, x) - solver.calculate_rho()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return super().predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return super().score(X, y)


class KernelSVC(LinearSVC, BiKernelSVC): #多分类核SVM
    def __init__(self,
                 C: float = 1.,
                 kernel: str = 'rbf',
                 degree: float = 3,
                 gamma: float = 'scale',
                 coef0: float = 0.,
                 max_iter: int = 1000,
                 rff: bool = False,
                 nystrom: bool = False,
                 D: int = 1000,
                 tol: float = 1e-5,
                 cache_size: int = 256,
                 shrinking: bool = False,
                 multiclass: str = "ovr",
                 n_jobs: int = None) -> None:
        super().__init__(C, max_iter, tol, cache_size, shrinking) #第一站：LinearSVC Python 首先根据 MRO 找到第一个父类 LinearSVC，并调用它的 __init__
                                                       #第二站：BiLinearSVC 如果 LinearSVC 的 __init__ 内部也写了 super().__init__(...)，它并不会跳到 object，而是会根据 MRO 找到下一个兄弟类，即 BiLinearSVC
                                                       #终点：object 最后，当所有父类都执行完，才会到达最顶层的 object
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.rff = rff
        self.nystrom = nystrom
        self.D = D
        params = {
            "estimator":
            BiKernelSVC(C, kernel, degree, gamma, coef0, max_iter, rff, nystrom, D, tol,
                        cache_size, shrinking),
            "n_jobs":
            n_jobs,
        }
        self.multiclass_model = {
            "ovo": OneVsOneClassifier(**params),
            "ovr": OneVsRestClassifier(**params),
        }[multiclass]

    #当你在 KernelSVC 中执行 super().fit() 时，Python 会按照以下顺序查找这个方法：
    #KernelSVC（当前类，没找到就往后走）
    #LinearSVC（在这里找到了！）
    #BiKernelSVC
    #BiLinearSVC
    #object
    def fit(self, X: np.ndarray, y: np.ndarray):
        return super().fit(X, y)

    def decision_function(self, X: np.ndarray):
        return super().decision_function(X)

    def predict(self, X: np.ndarray):
        return super().predict(X)

    def score(self, X: np.ndarray, y: np.ndarray):
        return super().score(X, y)


# class BiNuSVC(BiKernelSVC): #二分类NuSVM
#     def __init__(self,
#                  nu: float = 0.5,
#                  kernel: str = 'rbf',
#                  degree: float = 3.,
#                  gamma: float = 'scale',
#                  coef0: float = 0.,
#                  max_iter: int = 1000,
#                  rff: bool = False,
#                  D: int = 1000,
#                  tol: float = 1e-5,
#                  cache_size: int = 256) -> None:
#         super().__init__(1, kernel, degree, gamma, coef0, max_iter, rff, D,
#                          tol, cache_size)
#         self.nu = nu

#     def fit(self, X: np.ndarray, y: np.ndarray):
#         X, y = np.array(X), np.array(y, dtype=float)
#         y[y != 1] = -1
#         l, self.n_features = X.shape
#         p = np.zeros(l)

#         kernel_func = self.register_kernel(X.std())

#         def func(i):
#             return y * kernel_func(X, X[i:i + 1]).flatten() * y[i]

#         if self.cache_size == 0:
#             Q = y.reshape(-1, 1) * y * kernel_func(X, X)
#             solver = NuSolver(Q, p, y, self.nu * l, self.C, self.tol)
#         else:
#             solver = NuSolverWithCache(p, y, self.nu * l, self.C, func,
#                                        self.tol, self.cache_size)

#         for n_iter in range(self.max_iter):
#             i, j, Qi, Qj = solver.working_set_select(func)
#             if i < 0:
#                 break
#             solver.update(i, j, Qi, Qj)
#         else:
#             print("NuSVC not coverage with {} iterations".format(
#                 self.max_iter))

#         rho, b = solver.calculate_rho_b()
#         self.decision_function = lambda x: ((solver.alpha * y) @ kernel_func(X, x)) / rho + b / rho
#         return self

#     def predict(self, X: np.ndarray):
#         return super().predict(X)

#     def score(self, X: np.ndarray, y: np.ndarray):
#         return super().score(X, y)


# class NuSVC(KernelSVC, BiNuSVC): #多分类NuSVM
#     def __init__(self,
#                  nu: float = 0.5,
#                  kernel: str = 'rbf',
#                  degree: float = 3,
#                  gamma: float = 'scale',
#                  coef0: float = 0.,
#                  max_iter: int = 1000,
#                  rff: bool = False,
#                  D: int = 1000,
#                  tol: float = 1e-5,
#                  cache_size: int = 256,
#                  multiclass: str = "ovr",
#                  n_jobs: int = None) -> None:
#         super().__init__(1, kernel, degree, gamma, coef0, max_iter, rff, D,
#                          tol, cache_size, multiclass, n_jobs)
#         self.nu = nu
#         params = {
#             "estimator":
#             BiNuSVC(nu, kernel, degree, gamma, coef0, max_iter, rff, D, tol,
#                     cache_size),
#             "n_jobs":
#             n_jobs,
#         }
#         self.multiclass_model: OneVsOneClassifier = {
#             "ovo": OneVsOneClassifier(**params),
#             "ovr": OneVsRestClassifier(**params),
#         }[multiclass]

#     def fit(self, X: np.ndarray, y: np.ndarray):
#         return super().fit(X, y)

#     def predict(self, X: np.ndarray):
#         return super().predict(X)

#     def score(self, X: np.ndarray, y: np.ndarray):
#         return super().score(X, y)
