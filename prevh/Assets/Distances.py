from typing import Callable
import numpy as np
from prevh.Assets.Dataset.Dataset import Dataset
from prevh.Assets.Utils.Static import *
from prevh.Assets.Utils.ErrorHandler import ErrorHandler

class Distances:
    def __init__(self, algorithm: str, kwargs: dict | None = None):
        self.errorHandler = ErrorHandler()
        self.algorithm = algorithm
        self.function = self._get_func()
        self.kwargs = kwargs
        self._validate()

    def _validate(self):
        if self.algorithm not in VALID_DISTANCES_TYPES:
            self.errorHandler.exec(f"The distance algorithm type is not accepted."
                                   f"Valid types are: {VALID_DISTANCES_TYPES}.",
                                   "error", TypeError)
        if self.kwargs["p"] is None and self.algorithm == "minkowski":
            self.errorHandler.exec(f"The minkowski algorithm requires and p argument.",
                                   "error", TypeError)
        if self.algorithm == "mahalanobis":
            self.kwargs = {"inv_covariance_matrix": None}

    def _get_func(self) -> Callable:
        """Returns the distance function based on the chosen algorithm."""
        if self.algorithm == "euclidean":
            return self._euclidean
        if self.algorithm == "manhattan":
            return self._manhattan
        if self.algorithm == "minkowski":
            return self._minkowski
        if self.algorithm == "mahalanobis":
            return self._mahalanobis

    def _euclidean(self, ds: Dataset, target: np.ndarray) -> np.ndarray:
        """Efficient Euclidean distance computation."""
        # Vectorized Euclidean distance: sqrt(sum((x_i - y_i)^2))
        return np.sqrt(np.sum((ds.X - target) ** 2, axis=1))

    def _manhattan(self, ds: Dataset, target: np.ndarray) -> np.ndarray:
        """Efficient Manhattan distance computation."""
        # Vectorized Manhattan distance: sum(|x_i - y_i|)
        return np.sum(np.abs(ds.X - target), axis=1)

    def _minkowski(self, ds: Dataset, target: np.ndarray) -> np.ndarray:
        """Efficient Minkowski distance computation (with variable 'p')."""
        # Vectorized Minkowski distance: (sum(|x_i - y_i|^p))^(1/p)
        return np.sum(np.abs(ds.X - target) ** self.kwargs["p"], axis=1) ** (1 / self.kwargs["p"])

    def _mahalanobis(self, ds: Dataset, target: np.ndarray) -> np.ndarray:
        """Efficient Mahalanobis distance computation."""
        # Compute the covariance matrix only once and invert it
        if self.kwargs["covariance_matrix"] is None:
            cov_matrix = np.cov(ds.X, rowvar=False)
            self.kwargs["inv_covariance_matrix"] = np.linalg.inv(cov_matrix)

        # Efficient Mahalanobis distance using broadcasting
        diff = ds.X - target
        left_term = np.dot(diff, self.kwargs["inv_covariance_matrix"])
        return np.sqrt(np.sum(left_term * diff, axis=1))



