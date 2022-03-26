"""Prevh Classification Method"""
import numpy
import pandas as pd
import numpy as np
import math as mt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def prepareData(rawData, containsRelevance, *args):
    normalizedData, normalizedInput = [], []
    rawData.dropna(axis=0)  # cleaning dirty data into the training set before normalization
    if len(args) == 1:
        if containsRelevance: newRow = args[0] + [np.NaN, 0.0]
        else: newRow = args[0] + [np.NaN]
        newRow = pd.Series(newRow, index=rawData.columns)
        rawData = rawData.append(newRow, ignore_index=True)
    if containsRelevance:
        normalizedData = pd.DataFrame(MinMaxScaler(feature_range=(0, 1)).fit_transform(rawData[rawData.columns[:len(rawData.columns) - 2]]), columns=rawData.columns[:len(rawData.columns) - 2])
        normalizedData.insert(len(rawData.columns[:len(rawData.columns) - 2]), "label", rawData["label"], True)  # adding the label column to the normalizedData
        normalizedData.insert(len(rawData.columns[:len(rawData.columns) - 2]) + 1, "relevance", rawData["relevance"], True)  # adding the relevance column to the normalizedData
    else:
        try: normalizedData = pd.DataFrame(MinMaxScaler(feature_range=(0, 1)).fit_transform(rawData[rawData.columns[:len(rawData.columns) - 1]]), columns=rawData.columns[:len(rawData.columns) - 1])
        except: raise TypeError("Probably your data set does not contains relevance column.")
        normalizedData.insert(len(rawData.columns[:len(rawData.columns) - 1]), "label", rawData["label"], True)  # adding the label column to the normalizedData
    if len(args) == 1:
        normalizedInput = normalizedData.tail(1)
        normalizedData = normalizedData.dropna(axis=0)
        return normalizedData, normalizedInput
    else: return normalizedData

class PrevhClassifier:

    def __init__(self, data, **kwargs): # classifier initialization
        # KWARGS
        containsRelevance = kwargs.get("containsRelevance", False)
        # ERROR HANDLING IN PARAMETERS
        if not type(data) is pd.core.frame.DataFrame:
            raise TypeError("The training dataset must be a Pandas DataFrame object.")
        try:
            if containsRelevance and not data[data.columns[len(data.columns) - 1]].between(0, 1, inclusive="both").all():
                raise TypeError("At least one of the information relevance is not between 0 and 1.")
        except: raise TypeError("Relevance column does not exist in data set.")
        if containsRelevance:
            data.rename(columns={data.columns[len(data.columns) - 2]: 'label', data.columns[len(data.columns) - 1]: 'relevance'}, inplace=True)  # rename label and relevance header
        else:
            data.rename(columns={data.columns[len(data.columns) - 1]: 'label'}, inplace=True)  # rename label header
        # PrevhClassifier ATTRIBUTES DEFINITION
        self.rawData = data
        self.containsRelevance = containsRelevance
        if containsRelevance:
            self.features = data.columns[:len(data.columns) - 2]
            self.labels = data.iloc[:, len(data.columns) - 2].unique()
        else:
            self.features = data.columns[:len(data.columns) - 1]
            self.labels = data.iloc[:, len(data.columns) - 1].unique()
        self.rowsCount = data.shape[0]
        self.labelCount = data["label"].value_counts()
        self.normalizedData = prepareData(data, containsRelevance)

    def predict(self, *args, **kwargs):
        # CONSTANTS
        algorithms = ["KNN", "KNR"]  # possible space delimitation algorithms
        # FUNCTION GLOBAL VARIABLES
        train_set, unclassified_data, results = [], [], []
        # AUXILIARY FUNCTIONS
        def Euclidean_Dist(df1, df2, cols=self.features):
            return np.linalg.norm(df1[cols].values - df2[cols].values, axis=1)
        # KWARGS
        k = kwargs.get("k", self.rowsCount)
        algorithm = kwargs.get("algorithm", "KNR")
        # ERROR HANDLING IN PARAMETERS
        if len(args) > 2:
            raise TypeError("There are too many arguments. (Please verify the documentation file)")
        if algorithm not in algorithms:
            raise TypeError("The algorithm chosen is not supported. (Please verify the documentation file)")
        if algorithm == "KNR":
            for lab in self.labels:
                if k > self.normalizedData["label"].value_counts()[lab]:
                    raise TypeError("Does not exist enough values for {} using K = {}.".format(lab, k))
        # FUNCTION BEGIN
        if len(args) == 1: # Used for a simple predict
            train_set, unclassified_data = prepareData(self.rawData, self.containsRelevance, args[0])
        elif len(args) == 2: # Used in the score function
            train_set, unclassified_data = args[0].copy(), pd.DataFrame([args[1]], columns=self.normalizedData.columns)
        train_set["distance"] = Euclidean_Dist(train_set, unclassified_data)
        train_set = train_set.sort_values("distance")
        if algorithm == "KNN":
            if k > self.rowsCount:
                raise TypeError("k value can´t be greater than rows count.")
            train_set = train_set.head(k)
        if algorithm == "KNR":
            train_set_aux = pd.DataFrame(columns=self.normalizedData.columns)
            train_set = train_set.sort_values("label")
            for label in self.labels:
                train_set_aux = train_set_aux.append(train_set.head(k))
                train_set = train_set[train_set.label != label]
            train_set = train_set_aux
        if self.containsRelevance:
            train_set["relevance"] = train_set["relevance"].apply(lambda x: x * mt.sqrt(len(self.labels)))
            train_set["distance"] = train_set.apply(lambda x: (x["distance"] * x["relevance"]), axis=1)
        total_dist_sum = train_set["distance"].sum()
        for lab in self.labels:
            cur_label_dist = train_set.apply(lambda row: row["distance"] if row["label"] == lab else 0, axis=1).sum()
            if cur_label_dist != 0: results += [1 - (cur_label_dist / total_dist_sum)] # pertinence
            else: results += [np.NaN]
        results = pd.Series(results, index=self.labels).sort_values(ascending=False)
        return str(results.head(1).index[0])

    def calculateScore(self, scoreMethod, **kwargs):
        # CONSTANTS
        scoreMethods = ["TrainTestSplit", "KFold"]  # possible score methods
        algorithms = ["KNN", "KNR"]  # possible space delimitation algorithms
        # FUNCTION GLOBAL VARIABLES
        score = 0
        # KWARGS
        seed = kwargs.get("seed", None)
        algorithm = kwargs.get("algorithm", "KNR")
        if self.containsRelevance: labelIndex = -2
        else: labelIndex = -1
        # ERROR HANDLING IN PARAMETERS
        if scoreMethod not in scoreMethods:
            raise TypeError("The scoreMethod chosen is not supported. (Please verify the documentation file)")
        if algorithm not in algorithms:
            raise TypeError("The algorithm chosen is not supported. (Please verify the documentation file)")
        # FUNCTION BEGIN
        if scoreMethod == "TrainTestSplit":
            train_size = kwargs.get("train_size", 0.7)  # train data set percentage (TrainTestSplit)
            k = kwargs.get("k", int(train_size * self.rowsCount))
            correctPredictions = 0
            X_train, X_test, y_train, y_test = train_test_split(self.normalizedData, pd.DataFrame(self.normalizedData['label']), train_size=train_size, shuffle=True, random_state=seed)
            for i in X_test.to_numpy():
                if type(i[labelIndex]) is np.float64: cur_test_label = str(int(i[labelIndex]))
                else: cur_test_label = str(i[labelIndex])
                cur_predict = self.predict(X_train, i, k=k, algorithm=algorithm)
                if cur_test_label == cur_predict: correctPredictions += 1
            score = correctPredictions / X_test.shape[0] # calculate the score from correct prediction
        if scoreMethod == "KFold":
            n_splits = kwargs.get("n_splits", int(self.rowsCount / 2))  # number of division the data set will pass by (KFold)
            k = kwargs.get("k", int(self.rowsCount - (self.rowsCount / n_splits)))
            foldScore = []
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            indexFold = list(kf.split(self.normalizedData))
            for i in indexFold:
                correctPredictions = 0
                train_set = self.normalizedData.iloc[i[0]] # gets data frame from the index for the current fold
                test_set = self.normalizedData.iloc[i[1]]
                for y in test_set.to_numpy():
                    if type(y[labelIndex]) is np.float64: cur_test_label = str(int(y[labelIndex]))
                    else: cur_test_label = str(y[labelIndex])
                    cur_predict = self.predict(train_set, y, k=k, algorithm=algorithm)
                    if cur_test_label == cur_predict: correctPredictions += 1
                foldScore += [correctPredictions / test_set.shape[0]]
            score = sum(foldScore) / n_splits
        return score

# Some interesting results:
# Database - (scoreAlgorithm, spaceDelimitationMethod, k, train_size(TrainTestSplit)/n_splits(Kfold), randomnessSeed) => correct predict percentage
# IrisDataCSV - ("TrainTestSplit", algorithm="KNN", k=4, train_size=0.8, seed=42) => 0.966
# IrisDataCSV - ("TrainTestSplit", algorithm="KNR", k=25, train_size=0.8, seed=42) => 0.966
# IrisDataCSV - ("KFold", algorithm="KNR", k=40, n_splits=10, seed=42) => 0.946
