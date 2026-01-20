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

        # Model 1: RFF=True, cache_size=256
        print("Training: RFF=True, Cache=256...", end=" ", flush=True)
        custom_model = KernelSVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            max_iter=1000,
            rff=True,
            tol=1e-5,
            cache_size=256,
            multiclass="ovr",
            D=200,
        )
        custom_acc, custom_time, custom_memory, custom_warnings = run_with_metrics(
            custom_model, train_X, train_y, test_X, test_y, "RFF=True, Cache=256"
        )
        print(f"Done (warnings: {custom_warnings})")

        # Model 2: RFF=False, cache_size=256
        print("Training: RFF=False, Cache=256...", end=" ", flush=True)
        custom_model_with_cache = KernelSVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            max_iter=1000,
            rff=False,
            tol=1e-5,
            cache_size=256,
            multiclass="ovr",
            D=200,
        )
        cache_acc, cache_time, cache_memory, cache_warnings = run_with_metrics(
            custom_model_with_cache, train_X, train_y, test_X, test_y, "RFF=False, Cache=256"
        )
        print(f"Done (warnings: {cache_warnings})")

        # Shrinking comparison tests
        print("\n" + "-" * 100)
        print("Shrinking Effect Comparison:")
        print("-" * 100)

        # Model 5: shrinking=True, cache_size=256
        print("Training: Shrinking=True, Cache=256...", end=" ", flush=True)
        shrinking_cache_model = KernelSVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            max_iter=1000,
            rff=False,
            tol=1e-5,
            cache_size=256,
            multiclass="ovr",
            D=200,
            shrinking=True,
        )
        shrinking_cache_acc, shrinking_cache_time, shrinking_cache_memory, shrinking_cache_warnings = run_with_metrics(
            shrinking_cache_model, train_X, train_y, test_X, test_y, "Shrinking=True, Cache=256"
        )
        print(f"Done (warnings: {shrinking_cache_warnings})")

        # Note: The earlier "RFF=False, Cache=256" run uses shrinking=False (default),
        # so we reuse those results for the shrinking=False comparison.

        # Print results in a formatted table
        print("\nResults:")
        print(f"{'Model':<35s} | {'Accuracy':<10s} | {'Time (s)':<12s} | {'Memory (MB)':<15s} | {'Warnings':<10s}")
        print("-" * 100)
        print(f"{'RFF=True, Cache=256':<35s} | {custom_acc:>10.4f} | {custom_time:>12.3f} | {custom_memory:>15.2f} | {custom_warnings:>10d}")
        print(f"{'RFF=False, Cache=256':<35s} | {cache_acc:>10.4f} | {cache_time:>12.3f} | {cache_memory:>15.2f} | {cache_warnings:>10d}")
        
        print("\nShrinking Comparison:")
        print(f"{'Model':<35s} | {'Accuracy':<10s} | {'Time (s)':<12s} | {'Memory (MB)':<15s} | {'Warnings':<10s}")
        print("-" * 100)
        print(f"{'Shrinking=True, Cache=256':<35s} | {shrinking_cache_acc:>10.4f} | {shrinking_cache_time:>12.3f} | {shrinking_cache_memory:>15.2f} | {shrinking_cache_warnings:>10d}")
        print(f"{'Shrinking=False, Cache=256':<35s} | {cache_acc:>10.4f} | {cache_time:>12.3f} | {cache_memory:>15.2f} | {cache_warnings:>10d}")
        
        # Calculate speedup from shrinking
        print("\nShrinking Speedup Analysis:")
        print("-" * 100)
        if cache_time > 0:
            speedup_cache = cache_time / shrinking_cache_time if shrinking_cache_time > 0 else 0
            print(f"With Cache=256: Shrinking speedup = {speedup_cache:.2f}x (Time: {cache_time:.3f}s -> {shrinking_cache_time:.3f}s)")


if __name__ == "__main__":
    main()