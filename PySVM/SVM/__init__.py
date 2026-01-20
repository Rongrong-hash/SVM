"""
Expose SVM-related models.
"""

from .SVC import KernelSVC

try:
    from .SVCbaseline import SklearnSVCBaseline
    __all__ = ["KernelSVC", "SklearnSVCBaseline"]
except ImportError:
    __all__ = ["KernelSVC"]
