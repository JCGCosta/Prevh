# PrevhClassifier
This package implements the Prevh classification algorithm.
> The algorithm is based in the follow [research](https://zenodo.org/record/6090322#.Yj98bKbMKUk) **Pages 71-76**.
 
> [Package Documentation](https://pypi.org/project/prevhlib/).

> [![Publish to PyPI and TestPyPI](https://github.com/JCGCosta/Prevh/actions/workflows/python-publish.yml/badge.svg)](https://github.com/JCGCosta/Prevh/actions/workflows/python-publish.yml)

# User Guide

> This package can be installed with the following command: **pip install prevhlib**

## Python example:

```python
import numpy as np
import pandas as pd
from __init__.PrevhClassifier import PrevhClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

if __name__ == '__main__':
    iris = pd.read_csv('Datasets/iris.csv')

    X = iris.iloc[:, 0:4].values
    y = iris.iloc[:, 4].values
    header = (iris.columns[:-1], iris.columns[-1])

    prevh = PrevhClassifier(distance_algorithm="euclidean")
    prevh.fit(X, y, header=header, encoder=LabelEncoder(), scaler=StandardScaler())

    print(prevh)
    # Outputs: 
    # {  
    #    "dataset": {
    #       "header": {
    #          "features": "Index(['sepal length', 'sepal width', 'petal length', 'petal width'], dtype='object')",
    #          "classes": "class"
    #       },
    #       "encoder": "LabelEncoder()",
    #       "scaler": "StandardScaler()"
    #    },
    #    "distance": "euclidean"
    # }

    print(prevh.classify(np.array([5.1, 3.5, 1.4, 0.2]), K=3))
    # Outputs: (array(['Iris-setosa'], dtype=object), np.float64(0.2653212465045153))

    kfold_split_arguments = {
        "n_splits": 5,
        "random_state": 42,
        "shuffle": True
    }

    Evaluation_Results = prevh.evaluate(1, "kfold_cross_validation", kfold_split_arguments)

    print(Evaluation_Results.get_metrics())
    # Outputs:
    #       accuracy  precision    recall  f1-score
    # 0     0.966667   0.972222  0.962963  0.965899
    # 1     0.966667   0.969697  0.952381  0.958486
    # 2     0.966667   0.962963  0.966667  0.962848
    # 3     0.900000   0.911681  0.905556  0.907368
    # 4     0.966667   0.972222  0.972222  0.971014
    # Mean  0.953333   0.957757  0.951958  0.953123

    Evaluation_Results.plot_confusion_matrices()
```

<img src="https://raw.githubusercontent.com/JCGCosta/Prevh/refs/heads/main/confusion_matrix_example.png" width = "600">

## Next Steps

- In the next steps I will add support to other split, evaluation, encoder, and decoder methods.
- I expect to in new versions to comparisons between other machine learn method and the prevh classifier.
