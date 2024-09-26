import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

def apply_train_test_split(X: np.ndarray, kwargs) -> tuple:
    res = []
    indices = np.arange(len(X))
    train_indices, test_indices = train_test_split(indices, **kwargs)
    res.append([np.array(train_indices), np.array(test_indices)])
    return tuple(res)

def apply_kfold_split(X: np.ndarray, kwargs) -> tuple:
    kf = KFold(**kwargs)
    splits = kf.split(X)
    return splits

def apply_stratified_kfold_split(X: np.ndarray, y: np.ndarray, kwargs) -> tuple:
    kf = StratifiedKFold(**kwargs)
    splits = kf.split(X, y)
    return splits
