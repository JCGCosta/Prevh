import numpy as np

from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             r2_score,
                             mean_squared_error,
                             mean_absolute_error,
                             multilabel_confusion_matrix)

class Metrics:
    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def _calc_accuracy(self):
        return np.float64(accuracy_score(self.y_true, self.y_pred))

    def _calc_precision(self):
        return precision_score(self.y_true, self.y_pred, average='macro', zero_division=1)

    def _calc_recall(self):
        return recall_score(self.y_true, self.y_pred, average='macro', zero_division=1)

    def _calc_f1_score(self):
        return f1_score(self.y_true, self.y_pred, average='macro', zero_division=1)

    def _calc_r2_score(self):
        return r2_score(self.y_true, self.y_pred)

    def _calc_mse_score(self):
        return mean_squared_error(self.y_true, self.y_pred)

    def _calc_mae_score(self):
        return mean_absolute_error(self.y_true, self.y_pred)

    def gen_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        self.y_true = y_true
        self.y_pred = y_pred
        return np.array((self._calc_accuracy(),
                         self._calc_precision(),
                         self._calc_recall(),
                         self._calc_f1_score(),
                         self._calc_r2_score(),
                         self._calc_mse_score(),
                         self._calc_mae_score()))

    @staticmethod
    def gen_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
        return multilabel_confusion_matrix(y_true, y_pred)