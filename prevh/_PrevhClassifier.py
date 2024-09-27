import json

import numpy as np
import pandas as pd

from prevh.Assets.Dataset.Dataset import Dataset
from prevh.Assets.Distances import Distances
from prevh.Assets.Evaluator.Evaluation import Evaluation
from prevh.Assets.Evaluator.Metrics import Metrics
from prevh.Assets.Predict import predict


from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder

class PrevhClassifier:
    def __init__(self,
                 distance_algorithm: str = "euclidean",
                 kwargs = None):
        '''
        :param distance_algorithm: Algorithm to calculate the distance between the points.
               Supported algorithms: ["euclidean", "manhattan", "minkowski", "mahalanobis"]
        :param kwargs: Is used to pass additional arguments to the distance algorithm like the p for the Minkowski distance
        '''
        if kwargs is None: kwargs = {"p": 3}
        self.dataset = Dataset()
        self.distance = Distances(distance_algorithm, kwargs)
        self.metrics = Metrics()

    def __repr__(self):
        '''
        :return: The dataset as a dataframe.
        '''
        return json.dumps({
            "dataset": {
                "header": {
                    "features": self.dataset.header[0].__repr__(),
                    "classes": self.dataset.header[1]
                },
                "encoder": self.dataset.encoder.__repr__(),
                "scaler": self.dataset.scaler.__repr__(),
            },
            "distance": self.distance.algorithm
        }, indent=3)

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            header: tuple = None,
            encoder: None | LabelEncoder = LabelEncoder,
            scaler: None | StandardScaler | MinMaxScaler | RobustScaler = None):
        '''
        :param X: Numerical Features
        :param y: Class (Categorical or Numeric)
        :param header: The header of the dataset, like the feature names and class header
        :param encoder: The Class encoder to speed up the processing (LabelEncoder is the only supported by now)
                        Supported algorithms: ["LabelEncoder"]
        :param scaler: The Scaler used to re-scale the features to prevent long term convergences
                       Supported algorithms: ["StandardScalar", "MinMaxScaler", "RobustScaler"]
        :return: None
        '''
        self.dataset.fit(X=X, y=y, header=header, scaler=scaler, encoder=encoder)
        if scaler is not None: self.dataset.normalize_dataset()
        if encoder is not None: self.dataset.encode_y()

    def evaluate(self,
             K: int,
             split_algorithm: str,
             split_kwargs: dict) -> Evaluation:
        '''
        :param K: The number of K neighbors from each cluster to be analyzed.
        :param split_algorithm: Algorithm to split the data set for evaluation.
               Supported algorithms: ["train_test_split", "kfold_cross_validation"]
        :param split_kwargs: The kwargs for the split algorithm like the train and test sizes, random state etc.
               See the scikit-learn documentation to see the available arguments in:
               https://scikit-learn.org/stable/api/sklearn.model_selection.html
        :return: The dataframe matrix with the evaluation metrics.
                 The current metrics are Accuracy, AUC, Recall, Precision, F1
        '''
        # Each fold has a dataset split, and each dataset split is composed by (X_train, y_train, X_test, y_test)
        metrics = []
        confusion_matrix = []
        folds = self.dataset.split(split_algorithm, split_kwargs)
        for i, split in enumerate(folds):
            pred_y = []
            train_ds = Dataset()
            train_ds.fit(X=np.array(split[0]), y=np.array(split[1]))
            for test_X in split[2]:
                pred_y.append(predict(train_ds, self.distance, test_X, K)[0])
            metrics.append(self.metrics.gen_metrics(split[3], np.array(pred_y)))
            confusion_matrix.append(self.metrics.gen_confusion_matrix(split[3], np.array(pred_y)))
        metrics = pd.DataFrame(metrics, columns=["accuracy", "precision", "recall", "f1-score"])
        return Evaluation(metrics, confusion_matrix, self.dataset.get_labels())

    def classify(self,
                target: np.ndarray,
                K: int) -> [np.ndarray, np.float64]:
        '''
        :param target: The target unclassified entity to be predicted
        :param K: The number of K neighbors from each cluster to be analyzed
        :return: A list containing the predicted class and predicted class
        '''
        self.dataset.have_match_dimensions(target)
        if self.dataset.scaler is not None: target = self.dataset.normalize_target([target])
        min_key, min_value = predict(self.dataset, self.distance, target, K)
        return self.dataset.decode(min_key), min_value