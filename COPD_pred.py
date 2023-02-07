import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression

# Loads the dataset
data = pd.read_csv("dataset.csv")

# drops the irrelevant features
data = data.drop(['Unnamed: 0', 'ID'], axis=1)

# Replaces the text classifications for COPD severity with numerical ones
severity = data["COPDSEVERITY"]
severity = severity.replace({"MILD": 0, "MODERATE" : 1, "SEVERE" : 2, "VERY SEVERE" : 3})
data["COPDSEVERITY"] = severity

data_x = data.drop(["COPDSEVERITY"], axis=1)
data_y = data["COPDSEVERITY"]

print(data.head())