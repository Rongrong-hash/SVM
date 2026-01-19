import numpy as np
from functools import lru_cache

from scipy.stats import alpha


class Solver:
    def __init__(self,
                 Q: np.ndarray,
                 p: np.ndarray,
                 y: np.ndarray,
                 C: float,
                 tol: float = 1e-5) -> None:
        problem_size = p.shape[0]; #训练样本的总数量,.shape 返回各个维度
        assert problem_size == y.shape[0] #检查并确保待优化的目标向量 p 的规模必须与标签向量 y 的样本数量完全一致
        if Q is not None:
            assert problem_size == Q.shape[0]
            assert problem_size == Q.shape[1]

        self.Q = Q
        self.p = p
        self.y = y
        self.C = C
        self.tol = tol
        self.alpha = np.zeros(problem_size)

        self.neg_y_grad = -y * p #求 -y·▽f(α).对 {alpha} 求梯度（偏导数向量）nabla f({alpha}),在算法刚开始时（即 __init__ 阶段），所有的 alpha_i 都初始化为 0。此时 nabla f({alpha})_{alpha=0} = {p},所以，self.neg_y_grad 在初始时刻计算的就是 -y_i * nabla_i f({alpha}) = -y_i * p_i

    def working_set_select(self): #选择要改变的 alpha_i 与 alpha_j
        Iup = np.argwhere(
            np.logical_or(
                np.logical_and(self.alpha < self.C, self.y > 0),
                np.logical_and(self.alpha > 0, self.y < 0)
            )).flatten() #Iup 是 alpha 的下标
        Ilow = np.argwhere(
            np.logical_or(
                np.logical_and(self.alpha < self.C, self.y < 0),
                np.logical_and(self.alpha > 0, self.y > 0)
            )).flatten()

        find_fail = False
        try:
            i = Iup[np.argmax(self.neg_y_grad[Iup])] #求 Iup 中使 y_i * ▽f(α_i) 最小的下标 i
            j = Ilow[np.argmin(self.neg_y_grad[Ilow])] #求 Iup 中使 y_i * ▽f(α_i) 最大的下标 i
        except:
            find_fail = True #所有正类样本的 alpha 都已经是 C 了，且所有负类样本的 alpha 都已经是 0 了。此时 Iup, Ilow 就会找不到任何符合条件的样本

        if find_fail or self.neg_y_grad[i] - self.neg_y_grad[j] < self.tol:
            return -1, -1
        return i, j

    def update(self, i: int, j: int, func = None): #变量更新，在保证变量满足约束的条件下对 alpha_i 与 alpha_j 进行更新。约束条件：sum(y_i * alpha_i) = 0 and 0 <= alpha <= C
        Qi, Qj = self.get_Q(i, func), self.get_Q(j, func)
        yi, yj = self.y[i], self.y[j]
        ai_old, aj_old = self.alpha[i], self.alpha[j]

        quad_coef = Qi[i] + Qj[j] - 2 * yi * yj * Qi[j] #quad_coef 是 alpha_j^2 系数的2倍
        if quad_coef <= 0: #防止 quad_coef 为 0 时，除以 quad_coef 导致程序崩溃
            quad_coef = 1e-12

        if yi != yj: #意味着一个是正类 +1，一个是负类 -1。等式约束简化为：alpha_i - alpha_j = 常数
            L = max(0, aj_old - ai_old)
            H = min(self.C, self.C + aj_old - ai_old)
        else: #意味着两个都是正类 +1，两个都是负类 -1。等式约束简化为：alpha_i + alpha_j = 常数
            L = max(0, ai_old + aj_old - self.C)
            H = min(self.C, ai_old + aj_old)

        if L == H:
            return 0, 0

        if yi != yj: #计算 alpha_j 的更新步长
            delta = (self.neg_y_grad[i] * yi + self.neg_y_grad[j] * yj) / quad_coef
        else:
            delta = (self.neg_y_grad[j] * yj - self.neg_y_grad[i] * yi) / quad_coef

        aj_new = aj_old + delta #更新并剪枝 alpha_j
        if aj_new > H:
            aj_new = H
        elif aj_new < L:
            aj_new = L

        ai_new = ai_old + yi * yj * (aj_old - aj_new) #根据等式约束更新 alpha_i

        self.alpha[i] = ai_new
        self.alpha[j] = aj_new #更新状态
            
        delta_i = ai_new - ai_old
        delta_j = aj_new - aj_old
        self.neg_y_grad -= self.y * (delta_i * Qi + delta_j * Qj)
        return delta_i, delta_j

    def calculate_rho(self) -> float: #计算偏置项
        sv = np.logical_and( #自由支持向量对应的下标
            self.alpha > 0,
            self.alpha < self.C,
        )
        if sv.any() > 0: #有自由支持向量：取它们的平均值（最标准、最精确）
            rho = -np.average(self.neg_y_grad[sv])
        else: #无自由支持向量：取最严苛的下界最大值和上界最小值
            ub_id = np.logical_or( #-b 的上届
                np.logical_and(self.alpha == 0, self.y < 0),
                np.logical_and(self.alpha == self.C, self.y > 0)
            )
            lb_id = np.logical_or( #-b 的下届
                np.logical_and(self.alpha == 0, self.y > 0),
                np.logical_and(self.alpha == self.C, self.y < 0)
            )

            g_max = self.neg_y_grad[lb_id].max() if lb_id.any() else -np.inf
            g_min = self.neg_y_grad[ub_id].min() if ub_id.any() else np.inf #加上 np.any() 检查防止空集报错, np.inf 是正无穷

            if g_max != -np.inf and g_min != np.inf:
                rho = -(g_max + g_min) / 2
            else: #如果某一边为空，则取存在的另一边的极值
                rho = -g_max if g_max != -np.inf else -g_min

        return float(rho)

    def get_Q(self, i: int, func = None): #如果内存足够存储 Q，那么就直接返回 Q[i]
                                          #如果内存不足，可以通过 func(i) 现场计算 Q[i]
                                          #用 self.Q[i] 比较快
        return self.Q[i]


class SolverWithCache(Solver): #带核函数缓存机制的Solver
    def __init__(self,
                 p: np.ndarray,
                 y: np.ndarray,
                 C: float,
                 tol: float = 1e-5,
                 cache_size: int = 256) -> None:
        super().__init__(None, p, y, C, tol)

        self.cache_size = cache_size 
        self.get_Q = lru_cache(maxsize=self.cache_size)(self._get_Q_raw) # 在内存中开辟一块空间，把 get_Q 的返回结果（即 Q 矩阵的一整行）存起来
                                                                         #如果下一轮迭代又要用到第 i 行，装饰器会直接从内存里把这一行“扔”给程序，而不需要重新运行 func(i),点积运算被瞬间跳过
                                                                         #当缓存的行数超过 cache_size（比如 256 行）时，它会自动删掉最久没被用过的那一行，为新行腾出空间

    def working_set_select(self):
        return super().working_set_select()

    def update(self, i: int, j: int, func=None):
        return super().update(i, j, func)

    def calculate_rho(self):
        return super().calculate_rho()

    def _get_Q_raw(self, i, func):
        return func(i)


class NuSolver(Solver):
    def __init__(self, 
                 Q: np.ndarray, 
                 p: np.ndarray, 
                 y: np.ndarray, 
                 t: float, #nu * l
                 C: float, 
                 tol: float = 1e-5) -> None:
        super().__init__(Q, p, y, C, tol)
        problem_size = p.shape[0]
        assert problem_size == y.shape[0]
        if Q is not None:
            assert problem_size == Q.shape[0]
            assert problem_size == Q.shape[1]

        sum_pos = sum_neg = t / 2 #初始的 alpha 必须满足以下两个条件：
                                        #-等式约束（标签平衡）：sum(y_i * alpha_i) = 0
                                        #-nu 约束（总量限制）：sum(alpha_i) = nu * l = t
                                        #给正类样本分配一半的总权重（t/2），给负类样本也分配一半的总权重（t/2）
        self.alpha = np.empty(problem_size)

        for i in range(problem_size):  #-等式约束（标签平衡）：sum(y_i * alpha_i) = 0
                                       #-nu 约束（总量限制）：sum(alpha_i) = nu * l = t
                                       #-变量范围：0 <= alpha_i <= 1
            if self.y[i] == 1:
                self.alpha[i] = min(1., sum_pos)
                sum_pos -= self.alpha[i]
            else:
                self.alpha[i] = min(1., sum_neg)
                sum_neg -= self.alpha[i]

        self.neg_y_grad = -self.y * (np.matmul(Q, self.alpha) + self.p)
        self.QD = np.diag(self.Q) #矩阵 Q 的主对角线元素

    def working_set_select(self, func=None):
        Iup = np.argwhere(
            np.logical_or(
                np.logical_and(self.alpha < self.C, self.y > 0),
                np.logical_and(self.alpha > 0, self.y < 0),
            )).flatten()
        Ilow = np.argwhere(
            np.logical_or(
                np.logical_and(self.alpha < self.C, self.y < 0),
                np.logical_and(self.alpha > 0, self.y > 0),
            )).flatten()

        pos_fail, neg_fail = False, False
        try: #在正类 (y > 0) 中
            Imp = Iup[self.y[Iup] > 0] #alpha_i * y_i 增大的，y_i > 0,故为 alpha_i 增大的，delta(alpha_i) = delta
            IMp = Ilow[self.y[Ilow] > 0] #alpha_j * y_j 减小的，y_j > 0,故为 alpha_j 减小的，delta(alpha_j) = -delta
            #delta(f) = ▽f(α_i) * delta_i + ▽f(α_j) * delta_j = (▽f(α_i) - ▽f(α_j)) * delta
            #需要最小的 ▽f(α_i) 与最大的 ▽f(α_j)，由于 y_i = y_j = 1
            #需要最大的 -y_i * ▽f(α_i) 与最小的 -y_j * ▽f(α_j)
            i_p = Imp[np.argmax(self.neg_y_grad[Imp])]
            j_p = IMp[np.argmin(self.neg_y_grad[IMp])]
        except:
            pos_fail = True

        try: #在负类 (y < 0) 中
            Imn = Iup[self.y[Iup] < 0] #alpha_i * y_i 增大的，y_i < 0,故为 alpha_i 减小的，delta(alpha_i) = -delta
            IMn = Ilow[self.y[Ilow] < 0] #alpha_j * y_j 减小的，y_j < 0,故为 alpha_j 增大的，delta(alpha_j) = delta
            #delta(f) = ▽f(α_i) * delta_i + ▽f(α_j) * delta_j = (-▽f(α_i) + ▽f(α_j)) * delta
            #需要最大的 ▽f(α_i) 与最小的 ▽f(α_j)，由于 y_i = y_j = -1
            #需要最大的 -y_i * ▽f(α_i) 与最小的 -y_j * ▽f(α_j)
            i_n = Imn[np.argmax(self.neg_y_grad[Imn])]
            j_n = IMn[np.argmin(self.neg_y_grad[IMn])]
        except:
            neg_fail = True

        if pos_fail and neg_fail:
            return -1, -1, -1, -1
        elif pos_fail:
            return i_n, j_n, self.get_Q(i_n, func), self.get_Q(j_n, func)
        elif neg_fail:
            return i_p, j_p, self.get_Q(i_p, func), self.get_Q(j_p, func)
        else: #刚才已经在正类中选出了“最强一对” (i_p, j_p)，在负类中选出了“最强一对” (i_n, j_n)。现在的任务是：二选一，挑出哪一对对目标函数的贡献（下降量）最大
            grad_diff_p = self.neg_y_grad[i_p] - self.neg_y_grad[j_p]
            Q_ip = self.get_Q(i_p, func)
            quad_coef = self.QD[i_p] + self.QD[j_p] - 2 * Q_ip[j_p]
            if quad_coef <= 0:
                quad_coef = 1e-12 #SVM 使用的核矩阵 Q 必须是半正定的。这意味着对于任何一对 i, j，其二阶导数项 quad_coef 必须大于等于 0
            obj_diff_p = -grad_diff_p**2 / quad_coef

            grad_diff_n = self.neg_y_grad[i_n] - self.neg_y_grad[j_n]
            Q_in = self.get_Q(i_n, func)
            quad_coef = self.QD[i_n] + self.QD[j_n] - 2 * Q_in[j_n]
            if quad_coef <= 0:
                quad_coef = 1e-12
            obj_diff_n = -grad_diff_n**2 / quad_coef

            if obj_diff_p < obj_diff_n:
                return i_p, j_p, Q_ip, self.get_Q(j_p, func)
            return i_n, j_n, Q_in, self.get_Q(j_n, func)

    def update(self, i, j, Qi, Qj):
        ai_old, aj_old = self.alpha[i], self.alpha[j]

        quad_coef = Qi[i] + Qj[j] - 2 * Qi[j]
        if quad_coef <= 0:
            quad_coef = 1e-12

        delta = (self.neg_y_grad[i] - self.neg_y_grad[j]) / quad_coef

        s = ai_old + aj_old #约束：alpha_i + alpha_j = sum (常数)，且 0 <= alpha <= C
        L = max(0, s - self.C)
        H = min(self.C, s)

        aj_new = aj_old + delta
        if aj_new > H:
            aj_new = H
        elif aj_new < L:
            aj_new = L

        ai_new = s - aj_new

        self.alpha[i] = ai_new
        self.alpha[j] = aj_new

        delta_i = ai_new - ai_old
        delta_j = aj_new - aj_old

        if delta_i != 0 or delta_j != 0:
            self.neg_y_grad -= self.y * (delta_i * Qi + delta_j * Qj)

        return delta_i, delta_j

    def calculate_rho_b(self): #r1 = b - rho; r2 = b + rho
        pos_sv = np.logical_and(
            np.logical_and(self.alpha > 0, self.alpha < 1),
            self.y == 1,
        )
        if pos_sv.sum() == 0:
            try:
                r1_max = self.neg_y_grad[np.logical_and(self.alpha == 1, self.y == 1)].max()
                r1_min = self.neg_y_grad[np.logical_and(self.alpha == 0, self.y == 1)].min()
                r1 = (r1_max + r1_min) / 2
            except:
                r1 = 0
        else:
            r1 = np.average(self.neg_y_grad[pos_sv])

        neg_sv = np.logical_and(
            np.logical_and(self.alpha > 0, self.alpha < 1),
            self.y == -1,
        )
        if neg_sv.sum() == 0:
            try:
                r2_max = self.neg_y_grad[np.logical_and(self.alpha == 0, self.y == -1)].max()
                r2_min = self.neg_y_grad[np.logical_and(self.alpha == 1, self.y == -1)].min()
                r2 = (r2_max + r2_min) / 2
            except:
                r2 = 0
        else:
            r2 = np.average(self.neg_y_grad[neg_sv])

        rho = (r2 - r1) / 2
        b = (r1 + r2) / 2

        return rho, b


class NuSolverWithCache(NuSolver, SolverWithCache):
    def __init__(self,p, y, t, C, func, tol=1e-5, cache_size=256) -> None:
        self.p = p
        self.y = y
        self.C = C
        self.tol = tol
        self.cache_size = cache_size

        problem_size = p.shape[0]
        self.get_Q = lru_cache(maxsize=self.cache_size)(self._get_Q_raw)

        sum_pos = sum_neg = t / 2
        self.alpha = np.zeros(problem_size)

        for i in range(problem_size):
            if self.y[i] == 1:
                self.alpha[i] = min(1., sum_pos)
                sum_pos -= self.alpha[i]
            else:
                self.alpha[i] = min(1., sum_neg)
                sum_neg -= self.alpha[i]

        self.neg_y_grad = np.zeros(problem_size)
        QD = []
        for i in range(problem_size):
            Q_i = self.get_Q(i, func) #调用缓存装饰器获取核矩阵 Q 的第 i 行,并存进 LRU 缓存
            grad_val = Q_i @ self.alpha + self.p[i] #Q_i @ self.alpha：相当于 sum_j(Q_{ij} * alpha_j)
                                                    #self.p[i]：这是线性项（在 nu-SVC 中通常是 0，但为了兼容性保留）
            self.neg_y_grad[i] = -self.y[i] * grad_val
            
            QD.append(Q_i[i])
        
        self.QD = np.array(QD)
    
    def _get_Q_raw(self, i, func):
        return func(i)

    def update(self, i, j, Qi, Qj):
        return super().update(i, j, Qi, Qj)

    def working_set_select(self, func=None):
        return super().working_set_select(func)

    def calculate_rho_b(self):
        return super().calculate_rho_b()