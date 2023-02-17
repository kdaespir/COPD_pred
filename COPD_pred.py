import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression
from sklearn.linear_model import LogisticRegression


class COPD_project:
    def __init__(self) -> None:

        # Loads the dataset
        self.data = pd.read_csv("dataset.csv")

        # drops the irrelevant features
        self.data = self.data.drop(['Unnamed: 0', 'ID'], axis=1)

        self.data = self.data.dropna(axis=0)

        # Replaces the text classifications for COPD severity with numerical ones
        severity = self.data["COPDSEVERITY"]
        severity = severity.replace({"MILD": 0, "MODERATE" : 1, "SEVERE" : 2, "VERY SEVERE" : 3})
        self.data["COPDSEVERITY"] = severity

        self.data_x = self.data.drop(["COPDSEVERITY"], axis=1)
        self.data_y = self.data["COPDSEVERITY"]

    def data_splits(self):
        # splites the data into training and testing data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data_x, self.data_y, random_state=0, test_size=0.3)


    def chisq_feat_sel(self):
        # uses chi2 to select which features to include in model
        # this method returns all features to include in model.
        # methods for model selection will be explored
        chi2_feat = SelectKBest(chi2, k="all")
        best_feat = chi2_feat.fit_transform(self.data_x, self.data_y)

        imp_feats = chi2_feat.get_support()

        self.chi2_inclusion = []
        for bool, feat in zip(imp_feats, self.data_x.columns):
            if bool:
                self.chi2_inclusion.append(feat)

       
    def anova_feat_sel(self):
        # Computes the ANOVA F value for each of the features. The higher the score the better the feature is at explaining the outcomes of the dataset

        model = SelectKBest(f_classif, k="all")
        anova_feat = model.fit(self.data_x, self.data_y)
        

        scores = anova_feat.scores_
        anovaf_scores = list(zip(scores, self.data_x.columns))
        self.anovaf_incl = sorted(anovaf_scores, key= lambda tup: tup[0], reverse=True)

    def reg_feat_sel(self):
        # Uses linear regression to determine the f scores for each of the features in the dataset. 
        model = SelectKBest(f_regression, k="all")
        anova_feat = model.fit(self.data_x, self.data_y)
        

        scores = anova_feat.scores_
        reg_scores = list(zip(scores, self.data_x.columns))
        self.regf_incl = sorted(reg_scores, key= lambda tup: tup[0], reverse=True)
    
    def output_diagnostics(self):
        with open("copd_pred_diag.txt", "w") as f:
            f.writelines(f"Chi Square Inlcusion: {self.chi2_inclusion}\nANOVA Inclusion Rank: {self.anovaf_incl}\nRegression Inclusion Rank: {self.regf_incl}")


# if __name__ == "main":
x = COPD_project()
x.data_splits()
x.chisq_feat_sel()
x.anova_feat_sel()
x.reg_feat_sel()
x.output_diagnostics()