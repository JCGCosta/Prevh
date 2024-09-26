import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix

class Metrics:
    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def calc_accuracy(self):
        return np.float64(accuracy_score(self.y_true, self.y_pred))

    def calc_precision(self):
        return precision_score(self.y_true, self.y_pred, average='macro', zero_division=1)

    def calc_recall(self):
        return recall_score(self.y_true, self.y_pred, average='macro', zero_division=1)

    def calc_f1_score(self):
        return f1_score(self.y_true, self.y_pred, average='macro', zero_division=1)

    def gen_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        self.y_true = y_true
        self.y_pred = y_pred
        return np.array((self.calc_accuracy(), self.calc_precision(), self.calc_recall(), self.calc_f1_score()))

    @staticmethod
    def gen_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
        return multilabel_confusion_matrix(y_true, y_pred)