# Prevh
This package implements the Prevh classification algorithm.
> The algorithm is based in the follow [research](https://zenodo.org/record/6090322#.Yj98bKbMKUk) **Pages 71-76**.
 
> [Package Documentation](https://pypi.org/project/prevhlib/).

# User Guide

> This package can be installed with the following command: **pip install prevhlib**

## Dataset
***The file must be a CSV file and the header must be included.***

The columns must be in the following order:
* The features columns;
* The label column;
* The relevance column (Optional).

```text
feature1,feature2,feature3,label,relevance
10,10,10,Blue,1.0
15,15,15,Blue,1.0
20,20,20,Blue,1.0
45,45,45,Green,1.0
50,50,50,Green,1.0
55,55,55,Green,1.0
80,80,80,Red,1.0
85,85,85,Red,1.0
90,90,90,Red,1.0
```

## Python example:

```python
import prevh as ph
import pandas as pd
# Creates the classifier
prevhClass = PrevhClassifier(pd.read_csv("irisDataCSV.csv",","))
# Label recurrence in the dataset (Important to use KNR method)
print(prevhClass.labelCount)
# Rows count in the dataset (Important to use KNN method)
print(prevhClass.rowsCount)
# Calculate the dataset score using the TrainTestSplit and KFold Cross-Validation methods
TrainTestSplitScore = prevhclass.calculateScore("TrainTestSplit", algorithm="KNN", k=4, train_size=0.8, seed=42)
KfoldScore = prevhclass.calculateScore("KFold", algorithm="KNR", k=35, n_splits=15, seed=42)
print("TrainTestSplitScore:", TrainTestSplitScore)
print("KFoldScore:", KfoldScore)
```
