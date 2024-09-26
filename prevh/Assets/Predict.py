import numpy as np
from Assets.Dataset import Dataset
from Assets.Distances import Distances

def predict(dataset: Dataset,
            distance: Distances,
            target: np.ndarray,
            K: int) -> (int, np.float64):
    '''
    :param dataset: The dataset object already fitted
    :param distance: The distance object already fitted
    :param target: The target unclassified entity to be predicted
    :param K: The number of K neighbors from each cluster to be analyzed
    :return: A list containing the predicted class and predicted class
    '''
    distance_list = distance.function(dataset, target)

    result = {}
    for cls in np.unique(dataset.y):
        result[cls] = sorted(distance_list[dataset.y == cls])

    result = {key: sum(values[:K]) for key, values in result.items()}

    min_key, min_value = np.array(min(result.items(), key=lambda item: item[1]))

    return min_key, np.float64(min_value)