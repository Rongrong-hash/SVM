"""
Quick benchmark comparing different KernelSVC configurations.
Outputs accuracy, total runtime (fit + eval), and memory usage for several classic datasets.
"""
import os
import sys
import time
import tracemalloc
import numpy as np
from sklearn.datasets import (
    load_digits, make_classification
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PySVM.SVM.SVC import KernelSVC

np.random.seed(2022)

# Define datasets; will select those matching the target sample size
DATASETS = [
    ("digits", load_digits),
    ("synthetic_10000", lambda return_X_y=True: make_classification(
        n_samples=10000, n_features=20, n_informative=15, n_redundant=5,
        n_classes=3, random_state=2022, return_X_y=return_X_y
    )),
    ("synthetic_2000", lambda return_X_y=True: make_classification(
        n_samples=2000, n_features=20, n_informative=15, n_redundant=5,
        n_classes=3, random_state=2022, return_X_y=return_X_y
    )),
    ("synthetic_3000", lambda return_X_y=True: make_classification(
        n_samples=3000, n_features=20, n_informative=15, n_redundant=5,
        n_classes=3, random_state=2022, return_X_y=return_X_y
    )),
    ("synthetic_5000", lambda return_X_y=True: make_classification(
        n_samples=5000, n_features=30, n_informative=20, n_redundant=10,
        n_classes=4, random_state=2022, return_X_y=return_X_y
    )),
    ("synthetic_120k", lambda return_X_y=True: make_classification(
        n_samples=120000, n_features=20, n_informative=15, n_redundant=5,
        n_classes=3, random_state=2022, return_X_y=return_X_y
    )),
]

# Model variants to benchmark
CONFIGS = [
    ("Exact RBF (no sampling)", {"rff": False, "nystrom": False}),
    ("RFF approximation", {"rff": True, "nystrom": False}),
    ("Nyström approximation", {"rff": False, "nystrom": True}),
]


def get_dataset_size(loader):
    """Get the total number of samples in a dataset."""
    X, y = loader(return_X_y=True)
    return len(X)


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


def run_with_metrics(model, train_X, train_y, test_X, test_y, model_name=""):
    """Run model and return accuracy, total time, and peak memory usage."""
    # Start memory tracking
    tracemalloc.start()
    
    # Fit model (suppress convergence warnings by redirecting stdout temporarily)
    import io
    from contextlib import redirect_stdout
    
    start = time.perf_counter()
    # Capture print statements during training
    f = io.StringIO()
    with redirect_stdout(f):
        model.fit(train_X, train_y)
    fit_time = time.perf_counter() - start
    
    # Check if there were convergence warnings
    output = f.getvalue()
    convergence_warnings = output.count("not converge")
    
    # Get memory usage after fit
    current, peak = tracemalloc.get_traced_memory()
    fit_memory = peak / (1024 * 1024)  # Convert to MB
    
    # Evaluate model
    start = time.perf_counter()
    acc = model.score(test_X, test_y)
    eval_time = time.perf_counter() - start
    
    # Get final memory usage
    current, peak = tracemalloc.get_traced_memory()
    total_memory = peak / (1024 * 1024)  # Convert to MB
    
    tracemalloc.stop()
    
    return acc, fit_time + eval_time, total_memory, convergence_warnings


def main():
    print("=" * 100)
    print("KernelSVC Configuration Comparison")
    print("=" * 100)
    print()
    
    # Filter datasets to only include those with exactly target_samples samples
    filtered_datasets = []
    target_samples = 10000
    
    print(f"Selecting datasets with exactly {target_samples} samples...")
    for name, loader in DATASETS:
        try:
            size = get_dataset_size(loader)
            if size == target_samples:
                filtered_datasets.append((name, loader))
                print(f"  [OK]  {name}: {size} samples")
            else:
                print(f"  [SKIP]{name}: {size} samples (need {target_samples})")
        except Exception as e:
            print(f"  [ERR] {name}: Error loading dataset - {e}")
    
    if not filtered_datasets:
        print(f"\nNo datasets found with exactly {target_samples} samples!")
        return
    
    print(f"\nProcessing {len(filtered_datasets)} dataset(s) with {target_samples} samples...")
    print()
    
    for name, loader in filtered_datasets:
        print(f"\nDataset: {name}")
        print("=" * 100)
        train_X, test_X, train_y, test_y = prepare_dataset(loader)
        print(f"Training samples: {len(train_X)}, Test samples: {len(test_X)}, Classes: {len(np.unique(train_y))}")
        print("-" * 100)

        results = []

        for base_name, params in CONFIGS:
            for secorder in (False, True):
                secorder_label = "on" if secorder else "off"
                display_name = f"{base_name} | secorder={secorder_label}"
                print(f"Training: {display_name}, Cache=256...", end=" ", flush=True)
                model = KernelSVC(
                    C=1.0,
                    kernel="rbf",
                    gamma="scale",
                    max_iter=1000,
                    tol=1e-5,
                    cache_size=256,
                    multiclass="ovr",
                    D=200,
                    secorder=secorder,
                    **params,
                )
                acc, total_time, total_memory, warnings = run_with_metrics(
                    model, train_X, train_y, test_X, test_y, display_name
                )
                results.append((base_name, secorder_label, acc, total_time, total_memory, warnings))
                print(f"Done (warnings: {warnings})")

        # Print results in a formatted table
        print("\nResults (secorder off vs on):")
        print(f"{'Model':<35s} | {'secorder':<10s} | {'Accuracy':<10s} | {'Time (s)':<12s} | {'Memory (MB)':<15s} | {'Warnings':<10s}")
        print("-" * 110)
        for base_name, secorder_label, acc, total_time, total_memory, warnings in results:
            print(f"{base_name:<35s} | {secorder_label:<10s} | {acc:>10.4f} | {total_time:>12.3f} | {total_memory:>15.2f} | {warnings:>10d}")


if __name__ == "__main__":
    main()