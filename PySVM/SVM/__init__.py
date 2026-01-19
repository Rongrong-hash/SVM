"""
Expose SVM-related models.
"""

from .SVC import KernelSVC
from .SVCbaseline import SklearnSVCBaseline

__all__ = ["KernelSVC", "SklearnSVCBaseline"]
