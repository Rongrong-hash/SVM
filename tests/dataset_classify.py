from typing import Any
import time

import numpy as np
from PySVM.SVM.SVC import BiLinearSVC, LinearSVC, KernelSVC, NuSVC
from PySVM.SVM.SVCbaseline import SklearnSVCBaseline
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine #导入经典数据集鸢尾花 (Iris)、乳腺癌 (Breast Cancer)、手写数字 (Digits) 和葡萄酒 (Wine)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(2022)

DATASETS = [
    ("iris", load_iris),
    ("wine", load_wine),
    ("breast_cancer", load_breast_cancer),
    ("digits", load_digits),
]


def prepare_dataset(loader):
    X, y = loader(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, random_state=2022
    )
    scaler = StandardScaler().fit(train_X)
    return (
        scaler.transform(train_X),
        scaler.transform(test_X),
        train_y,
        test_y,
    )


def run_with_time(model, train_X, train_y, test_X, test_y):
    start = time.perf_counter()
    model.fit(train_X, train_y)
    fit_time = time.perf_counter() - start

    start = time.perf_counter()
    acc = model.score(test_X, test_y)
    eval_time = time.perf_counter() - start
    return acc, fit_time + eval_time


# ---------------------------------------------------------------------------
# Baseline comparison: custom KernelSVC vs sklearn's SVC
# ---------------------------------------------------------------------------
print("Custom KernelSVC vs sklearn SVC (RBF)")
custom_acc = np.zeros((1, len(DATASETS)))
baseline_acc = np.zeros((1, len(DATASETS)))
custom_time = np.zeros((1, len(DATASETS)))
baseline_time = np.zeros((1, len(DATASETS)))

for i, (_, loader) in enumerate(DATASETS):
    train_X, test_X, train_y, test_y = prepare_dataset(loader)

    custom_model = KernelSVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        max_iter=1000,
        tol=1e-5,
        cache_size=256,
        multiclass="ovr",
    )
    custom_acc[0, i], custom_time[0, i] = run_with_time(
        custom_model, train_X, train_y, test_X, test_y
    )

    baseline_model = SklearnSVCBaseline(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        tol=1e-5,
        max_iter=1000,
    )
    baseline_acc[0, i], baseline_time[0, i] = run_with_time(
        baseline_model, train_X, train_y, test_X, test_y
    )

print("custom accuracy:")
print(custom_acc)
print("custom time (s):")
print(custom_time)
print("sklearn accuracy:")
print(baseline_acc)
print("sklearn time (s):")
print(baseline_time)


score = np.zeros((1, 4)) #创建一个 1x4 的矩阵，用于存放四个数据集的准确率
for i, load_dataset in enumerate(
    [load_iris, load_wine, load_breast_cancer, load_digits]):
    X, y = load_dataset(return_X_y=True) #return_X_y=True 使函数不再返回冗余信息。直接返回两个 Numpy 数组，第一个赋值给 X，第二个赋值给 y
    train_X, test_X, train_y, test_y = train_test_split(X, y) #切分 train_X (训练集特征)，train_y (训练集标签)，test_X (测试集特征)，test_y (测试集答案)
    stder = StandardScaler().fit(train_X) #StandardScaler()：创建一个"标准化器"工具。.fit(train_X)：让这个工具去"观察"训练集特征 train_X，并计算出两个关键的数学值：均值 (mu) 与标准差 (sigma)
    train_X = stder.transform(train_X) #使用上一步得到的训练集的 train_X 的均值与方差来将训练集 train_X 的均值与方差标准化为 0 与 1
    test_X = stder.transform(test_X) #使用 train_X 的均值与方差来处理 test_X

    model = BiLinearSVC(
        C=1.0,
        max_iter=1000,
        tol=1e-5,
        cache_size=256
    )
    score[0, i] = model.fit(train_X, train_y).score(test_X, test_y)

print("BiLinearSVC scores:")
print(score)

# Test LinearSVC with OVR strategy
print("\nLinearSVC (OVR) scores:")
linear_score_ovr = np.zeros((1, 4))
for i, load_dataset in enumerate(
    [load_iris, load_wine, load_breast_cancer, load_digits]):
    X, y = load_dataset(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=2022)
    stder = StandardScaler().fit(train_X)
    train_X = stder.transform(train_X)
    test_X = stder.transform(test_X)

    model = LinearSVC(
        C=1.0,
        max_iter=1000,
        tol=1e-5,
        cache_size=256,
        multiclass="ovr"
    )
    linear_score_ovr[0, i] = model.fit(train_X, train_y).score(test_X, test_y)

print(linear_score_ovr)

# Test LinearSVC with OVO strategy
print("\nLinearSVC (OVO) scores:")
linear_score_ovo = np.zeros((1, 4))
for i, load_dataset in enumerate(
    [load_iris, load_wine, load_breast_cancer, load_digits]):
    X, y = load_dataset(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=2022)
    stder = StandardScaler().fit(train_X)
    train_X = stder.transform(train_X)
    test_X = stder.transform(test_X)

    model = LinearSVC(
        C=1.0,
        max_iter=1000,
        tol=1e-5,
        cache_size=256,
        multiclass="ovo"
    )
    linear_score_ovo[0, i] = model.fit(train_X, train_y).score(test_X, test_y)

print(linear_score_ovo)

# Test KernelSVC with RBF kernel and OVR strategy
print("\nKernelSVC (RBF, OVR) scores:")
kernel_score_rbf_ovr = np.zeros((1, 4))
for i, load_dataset in enumerate(
    [load_iris, load_wine, load_breast_cancer, load_digits]):
    X, y = load_dataset(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=2022)
    stder = StandardScaler().fit(train_X)
    train_X = stder.transform(train_X)
    test_X = stder.transform(test_X)

    model = KernelSVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        max_iter=1000,
        tol=1e-5,
        cache_size=256,
        multiclass="ovr"
    )
    kernel_score_rbf_ovr[0, i] = model.fit(train_X, train_y).score(test_X, test_y)

print(kernel_score_rbf_ovr)

# Test KernelSVC with RBF kernel and OVO strategy
print("\nKernelSVC (RBF, OVO) scores:")
kernel_score_rbf_ovo = np.zeros((1, 4))
for i, load_dataset in enumerate(
    [load_iris, load_wine, load_breast_cancer, load_digits]):
    X, y = load_dataset(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=2022)
    stder = StandardScaler().fit(train_X)
    train_X = stder.transform(train_X)
    test_X = stder.transform(test_X)

    model = KernelSVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        max_iter=1000,
        tol=1e-5,
        cache_size=256,
        multiclass="ovo"
    )
    kernel_score_rbf_ovo[0, i] = model.fit(train_X, train_y).score(test_X, test_y)

print(kernel_score_rbf_ovo)

# Test KernelSVC with polynomial kernel and OVR strategy
print("\nKernelSVC (Poly, OVR) scores:")
kernel_score_poly_ovr = np.zeros((1, 4))
for i, load_dataset in enumerate(
    [load_iris, load_wine, load_breast_cancer, load_digits]):
    X, y = load_dataset(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=2022)
    stder = StandardScaler().fit(train_X)
    train_X = stder.transform(train_X)
    test_X = stder.transform(test_X)

    model = KernelSVC(
        C=1.0,
        kernel='poly',
        degree=3,
        gamma='scale',
        coef0=0.0,
        max_iter=1000,
        tol=1e-5,
        cache_size=256,
        multiclass="ovr"
    )
    kernel_score_poly_ovr[0, i] = model.fit(train_X, train_y).score(test_X, test_y)

print(kernel_score_poly_ovr)

# Test KernelSVC with RFF (Random Fourier Features) and OVR strategy
print("\nKernelSVC (RBF with RFF, OVR) scores:")
kernel_score_rff_ovr = np.zeros((1, 4))
for i, load_dataset in enumerate(
    [load_iris, load_wine, load_breast_cancer, load_digits]):
    X, y = load_dataset(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=2022)
    stder = StandardScaler().fit(train_X)
    train_X = stder.transform(train_X)
    test_X = stder.transform(test_X)

    model = KernelSVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        max_iter=1000,
        tol=1e-5,
        cache_size=256,
        rff=True,
        D=500,
        multiclass="ovr"
    )
    kernel_score_rff_ovr[0, i] = model.fit(train_X, train_y).score(test_X, test_y)

print(kernel_score_rff_ovr)

# Test NuSVC with RBF kernel and OVR strategy
print("\nNuSVC (RBF, OVR) scores:")
nu_score_rbf_ovr = np.zeros((1, 4))
for i, load_dataset in enumerate(
    [load_iris, load_wine, load_breast_cancer, load_digits]):
    X, y = load_dataset(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=2022)
    stder = StandardScaler().fit(train_X)
    train_X = stder.transform(train_X)
    test_X = stder.transform(test_X)

    model = NuSVC(
        nu=0.5,
        kernel='rbf',
        gamma='scale',
        max_iter=1000,
        tol=1e-5,
        cache_size=256,
        multiclass="ovr"
    )
    nu_score_rbf_ovr[0, i] = model.fit(train_X, train_y).score(test_X, test_y)

print(nu_score_rbf_ovr)

# Test NuSVC with RBF kernel and OVO strategy
print("\nNuSVC (RBF, OVO) scores:")
nu_score_rbf_ovo = np.zeros((1, 4))
for i, load_dataset in enumerate(
    [load_iris, load_wine, load_breast_cancer, load_digits]):
    X, y = load_dataset(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=2022)
    stder = StandardScaler().fit(train_X)
    train_X = stder.transform(train_X)
    test_X = stder.transform(test_X)

    model = NuSVC(
        nu=0.5,
        kernel='rbf',
        gamma='scale',
        max_iter=1000,
        tol=1e-5,
        cache_size=256,
        multiclass="ovo"
    )
    nu_score_rbf_ovo[0, i] = model.fit(train_X, train_y).score(test_X, test_y)

print(nu_score_rbf_ovo)
