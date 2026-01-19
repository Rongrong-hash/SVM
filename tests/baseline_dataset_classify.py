"""
Quick benchmark comparing the custom KernelSVC with sklearn's native SVC.
Outputs accuracy and total runtime (fit + eval) for several classic datasets.
"""
import os
import sys
import time
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure project root is on sys.path when executed directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PySVM.SVM.SVC import KernelSVC
from PySVM.SVM.SVC import NuSVC
from PySVM.SVM.SVCbaseline import SklearnSVCBaseline, SklearnNuSVCBaseline

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


def main():
    print("Custom KernelSVC vs sklearn SVC (RBF)")
    for name, loader in DATASETS:
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
        custom_acc, custom_time = run_with_time(
            custom_model, train_X, train_y, test_X, test_y
        )

        baseline_model = SklearnSVCBaseline(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            tol=1e-5,
            max_iter=1000,
        )
        baseline_acc, baseline_time = run_with_time(
            baseline_model, train_X, train_y, test_X, test_y
        )

        print(
            f"{name:13s} | custom acc: {custom_acc:.4f}, time: {custom_time:.3f}s "
            f"| sklearn acc: {baseline_acc:.4f}, time: {baseline_time:.3f}s"
        )

    print("\nCustom NuSVC vs sklearn NuSVC (RBF)")
    for name, loader in DATASETS:
        train_X, test_X, train_y, test_y = prepare_dataset(loader)

        custom_model = NuSVC(
            nu=0.5,
            kernel="rbf",
            gamma="scale",
            max_iter=1000,
            tol=1e-5,
            cache_size=256,
            multiclass="ovr",
        )
        custom_acc, custom_time = run_with_time(
            custom_model, train_X, train_y, test_X, test_y
        )

        baseline_model = SklearnNuSVCBaseline(
            nu=0.5,
            kernel="rbf",
            gamma="scale",
            tol=1e-5,
            max_iter=1000,
        )
        baseline_acc, baseline_time = run_with_time(
            baseline_model, train_X, train_y, test_X, test_y
        )

        print(
            f"{name:13s} | custom acc: {custom_acc:.4f}, time: {custom_time:.3f}s "
            f"| sklearn acc: {baseline_acc:.4f}, time: {baseline_time:.3f}s"
        )


if __name__ == "__main__":
    main()