import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression

# Loads the dataset
data = pd.read_csv("dataset.csv")
print(data.head())