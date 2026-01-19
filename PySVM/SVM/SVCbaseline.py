import numpy as np
from sklearn.base import BaseEstimator
from sklearn.svm import SVC, NuSVC


class SklearnSVCBaseline(BaseEstimator):
    """
    Thin wrapper around sklearn.svm.SVC.
    Acts as a baseline to compare with the custom implementation in SVC.py.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        # Keep kwargs so tests can mirror the custom model's params.
        self._kwargs = kwargs
        self.model: SVC | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model = SVC(**self._kwargs)
        self.model.fit(X, y)
        return self

    def decision_function(self, X: np.ndarray):
        if self.model is None:
            raise RuntimeError("Call fit before decision_function.")
        return self.model.decision_function(X)

    def predict(self, X: np.ndarray):
        if self.model is None:
            raise RuntimeError("Call fit before predict.")
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray):
        if self.model is None:
            raise RuntimeError("Call fit before score.")
        return self.model.score(X, y)


class SklearnNuSVCBaseline(BaseEstimator):
    """
    Thin wrapper around sklearn.svm.NuSVC.
    Acts as a baseline to compare with the custom NuSVC implementation in SVC.py.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._kwargs = kwargs
        self.model: NuSVC | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model = NuSVC(**self._kwargs)
        self.model.fit(X, y)
        return self

    def decision_function(self, X: np.ndarray):
        if self.model is None:
            raise RuntimeError("Call fit before decision_function.")
        return self.model.decision_function(X)

    def predict(self, X: np.ndarray):
        if self.model is None:
            raise RuntimeError("Call fit before predict.")
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray):
        if self.model is None:
            raise RuntimeError("Call fit before score.")
        return self.model.score(X, y)