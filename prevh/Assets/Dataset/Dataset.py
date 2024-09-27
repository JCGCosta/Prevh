import pandas as pd

from prevh.Assets.Utils.ErrorHandler import ErrorHandler
from prevh.Assets.Dataset.Split import *

class Dataset:
    def __init__(self):
        self.X = None
        self.y = None
        self.header = None
        self.scaler = None
        self.encoder = None
        self.errorHandler = ErrorHandler()

    def _validate(self):
        if not isinstance(self.X, np.ndarray):
            self.errorHandler.exec("The features are not from np.ndarray type.","error", TypeError)

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            scaler = None,
            encoder = None,
            header: tuple | None = None):
        self.X = X
        self.y = y
        self.header = header
        self.scaler = scaler
        self.encoder = encoder
        self._validate()

    def have_match_dimensions(self, target: np.ndarray):
        if not all(i.shape == target.shape for i in self.X):
            self.errorHandler.exec("Target doesnt match the features dimensions.", "error", ValueError)

    def encode_y(self):
        self.y = self.encoder.fit_transform(self.y)

    def decode(self, cls_code: int) -> np.ndarray:
        return self.encoder.inverse_transform([int(cls_code)])

    def get_labels(self) -> list:
        return self.encoder.classes_

    def normalize_dataset(self):
        self.X = self.scaler.fit_transform(self.X)

    def normalize_target(self, target):
        return self.scaler.transform(target)

    def get_df(self):
        df = pd.DataFrame(self.X, columns=self.header[0])
        df[self.header[1]] = self.y
        return df

    def split(self, split_algorithm: str, kwargs: dict) -> list:
        splits = []
        results = []
        if split_algorithm == "train_test_split":
            splits = apply_train_test_split(self.X, kwargs)
        if split_algorithm == "kfold_cross_validation":
            splits = apply_kfold_split(self.X, kwargs)
        if split_algorithm == "stratified_kfold_cross_validation":
            splits = apply_stratified_kfold_split(self.X, self.y, kwargs)
        for i, split in enumerate(splits):
            train_X = tuple(self.X[i] for i in split[0])
            test_X = tuple(self.X[i] for i in split[1])
            train_y = tuple(self.y[i] for i in split[0])
            test_y = tuple(self.y[i] for i in split[1])
            results.append((train_X, train_y, test_X, test_y))
        return results