import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression


class COPD_project:
    def __init__(self) -> None:

        # Loads the dataset
        data = pd.read_csv("dataset.csv")

        # drops the irrelevant features
        data = data.drop(['Unnamed: 0', 'ID'], axis=1)

        data = data.dropna(axis=0)

        # Replaces the text classifications for COPD severity with numerical ones
        severity = data["COPDSEVERITY"]
        severity = severity.replace({"MILD": 0, "MODERATE" : 1, "SEVERE" : 2, "VERY SEVERE" : 3})
        self.data["COPDSEVERITY"] = severity

        self.data_x = data.drop(["COPDSEVERITY"], axis=1)
        self.data_y = data["COPDSEVERITY"]

    def data_splits(self):
        # splites the data into training and testing data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data_x, self.data_y, random_state=0, test_size=0.3)


    def chisq_feat_sel(self):
        # uses chi2 to select which features to include in model
        # this method returns all features to include in model.
        # methods for model selection will be explored
        chi2_feat = SelectKBest(chi2, k="all")
        best_feat = chi2_feat.fit_transform(self.data_x, self.data_y)

        mask = chi2_feat.get_support()

        inclusion = []
        for bool, feat in zip(mask, self.data_x.columns):
            if bool:
                inclusion.append(feat)
                
        print(len(inclusion), len(self.data_x.columns))