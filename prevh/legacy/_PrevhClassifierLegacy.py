"""Prevh Classification"""

import pandas as pd
import numpy as np
import math as mt
from sklearn.preprocessing import MinMaxScaler

class PrevhClassifier:

    def __init__(self, df_dataset, **kwargs): # classifier initialization
        # train test verifications
        if not type(df_dataset) is pd.core.frame.DataFrame:
            raise TypeError("First argument must be a Pandas DataFrame object.")
        if not df_dataset[df_dataset.columns[len(df_dataset.columns) - 1]].between(0, 1, inclusive="both").all():
            raise TypeError("At least one of the information relevance is not between 0 and 1.")
        # classifier creation method
        self.rawdata = df_dataset
        self.spacedelimitationmethod = kwargs.get('delimitationMethod', "KNN")
        self.axisheader = df_dataset.columns[:len(df_dataset.columns) - 2]
        self.posibleresults = df_dataset.iloc[:, len(df_dataset.columns) - 2].unique()
        self.datacount = df_dataset.shape[0]
        self.resultsheader = df_dataset.columns[len(df_dataset.columns) - 2]
        self.relevationheader = df_dataset.columns[len(df_dataset.columns) - 1]


    def predict_pertinence(self, inputlist, **kwargs): # classifier pertinence prediction
        # n-dimensional euclidean distance function
        def Euclidean_Dist(df1, df2, cols=self.axisheader):
            return np.linalg.norm(df1[cols].values - df2[cols].values, axis=1)

        # Kwargs default definitions and return list creation
        list_of_predict_results = []
        nneighbors = kwargs.get('k', self.datacount)
        # test set verifications
        if isinstance(nneighbors, int):
            nneighbors = np.full(shape=len(inputlist), fill_value=nneighbors).tolist()
        if not isinstance(inputlist, list):
            raise TypeError("The prediction input parameter must be a list of lists.")
        if not isinstance(nneighbors, list):
            raise TypeError("The nNeighbors parameter must be a list.")
        if len(nneighbors) != len(inputlist):
            raise TypeError("The input and nNeighbors parameters must have the same length.")
        # Starts Prediction
        for e, i in enumerate(inputlist):
            # test set verifications
            if not isinstance(i, list):
                raise TypeError("The prediction input must be a list of lists.")
            if len(i) != len(self.axisheader):
                raise TypeError("At least one of the prediction inputs does not match with axis length.")
            if nneighbors[e] > self.datacount or nneighbors[e] < 1 or not isinstance(nneighbors[e], int):
                raise TypeError("The nNeighbors parameter must be an integer and be in between 1 and " + str(self.datacount) + " (inclusive).")
            predict_data = self.rawdata.copy()
            predict_results = pd.DataFrame(columns=[self.resultsheader, "pertinence", "relevance"])
            predict_data.loc[len(predict_data)] = i + [None, "0"]
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaleddata = scaler.fit_transform(predict_data[self.axisheader])
            predict_data = pd.DataFrame(scaleddata, columns=self.axisheader)
            predict_data.insert(len(self.axisheader), self.resultsheader, self.rawdata[self.resultsheader], True)
            predict_data.insert(len(self.axisheader) + 1, self.relevationheader, self.rawdata[self.relevationheader], True)
            predict_input = predict_data.loc[pd.isna(predict_data[self.resultsheader])]
            predict_input = (predict_input.drop([self.resultsheader, self.relevationheader], axis=1)).reset_index(drop=True)
            predict_data = predict_data.loc[pd.notna(predict_data[self.resultsheader])]
            print(predict_data)
            print(predict_input)
            predict_data["distance"] = Euclidean_Dist(predict_data, predict_input)
            predict_data = predict_data.sort_values("distance").reset_index(drop=True)
            if self.spacedelimitationmethod == "KNN": predict_data = predict_data[predict_data.index < nneighbors[e]]
            #if self.spacedelimitationmethod == "KNR": #PROGRAMAR KNR E TESTAR
            predict_data[self.relevationheader] = predict_data[self.relevationheader].apply(lambda x: x * mt.sqrt(len(self.axisheader)))
            predict_data["powered distance"] = predict_data.apply(lambda x: (x["distance"] * x[self.relevationheader]), axis=1)
            powered_dist_sum = predict_data["powered distance"].sum()
            for p, r in enumerate(self.posibleresults):
                subset_df = predict_data[predict_data[self.resultsheader] == r]
                column_sum = subset_df["powered distance"].sum()
                if column_sum != 0:
                    predict_results.loc[p] = [r, 1 - (column_sum/predict_data["powered distance"].sum()), mt.sqrt(len(self.axisheader)) * (1 - (column_sum/predict_data["powered distance"].sum()))]
            list_of_predict_results += [[predict_results, e]]
        return list_of_predict_results

PrevhClassifier(pd.read_csv("../dataSetExamples/dataWithRelevance.csv", ",")).predict_pertinence([[10, 10, 10], [20, 20, 20]], k=6)