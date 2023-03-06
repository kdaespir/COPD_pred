import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

def copd_group_labeler(df,datasets):
    if df.equals(datasets[0]):
        return "Mild"
    if df.equals(datasets[1]):
        return "Moderate"
    if df.equals(datasets[2]):
        return "Severe"
    if df.equals(datasets[3]):
        return "Very Severe"

class COPD_project:
    def __init__(self) -> None:

        # Loads the dataset
        self.data = pd.read_csv("dataset.csv")

        # drops the irrelevant features
        self.data = self.data.drop(['Unnamed: 0', 'ID'], axis=1)

        # drops any rows that possess a na value
        self.data = self.data.dropna(axis=0)

        # checks for any duplicates in the data, if so it drops them by the row
        self.data = self.data.drop_duplicates()

        # Replaces the text classifications for COPD severity with numerical ones
        severity = self.data["COPDSEVERITY"]
        severity = severity.replace({"MILD": 0, "MODERATE" : 1, "SEVERE" : 2, "VERY SEVERE" : 3})
        self.data["COPDSEVERITY"] = severity


        # This creates a new feature which is the FEV value divided by the FVC value as this 
        fev_fvc_ratio = self.data["FEV1"] / self.data["FVC"]
        self.data["fev_fvc_ratio"] = fev_fvc_ratio

        # splits the data into its x and y components used when predicting COPD severity
        self.data_x = self.data.drop(["COPDSEVERITY"], axis=1)
        self.data_y = self.data["COPDSEVERITY"]

        self.data.to_csv("PS627_processed_data.csv")
        self.mode = int(input("Feature Selection or Model execution (0/1) "))

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
        # outputs the feature selection diagnostics to a seperate text file for storage and greater readability
        with open("copd_pred_diag.txt", "w") as f:
            f.writelines(f"Chi Square Inlcusion: {self.chi2_inclusion}\nANOVA Inclusion Rank: {self.anovaf_incl}\nRegression Inclusion Rank: {self.regf_incl}")


    def copd_pred_logistic(self):
        # splits the data into dependant and independant features (x and y respectively)
        func_abx = self.data[["FEV1PRED", "FEV1", "FVCPRED", "FVC","fev_fvc_ratio","MWT2"]]
        func_aby = self.data["COPDSEVERITY"]

        # splits the x and y data into training and testing data that will be used to fit the logistic regression model, 30% of the data is
        # reserved for the testing data
        xtrain, xtest, ytrain, ytest = train_test_split(func_abx, func_aby, random_state=0, test_size=0.3)

        # sets the model to a logistic regression model with a max itreation to 1000 to ensure that the model converges
        model = LogisticRegression(max_iter=1000)
        # fits the model with the training x and y data
        model.fit(xtrain, ytrain)
        # makes predictions based on the x testing data
        pred = model.predict(xtest)

        # gets the r2 and mse value for the for the predictions 
        model_r2 = r2_score(ytest, pred)
        model_mse = mean_squared_error(ytest, pred)
        # determines the accuracy score based on the predictions and the true y values
        acc_score = accuracy_score(ytest, pred)

        # prints the performance metrics of the logistic regression model 
        print(f"the r2 score is {model_r2} whereas the mse is {model_mse}, the accuracy is {acc_score}")

    def slr(self, metric):
        # Performs a single linear regression using metric as the independant feature in the regression

        #Specifies the features to be assesed by the OLS regession
        features = ["MWT1Best", "FEV1", "FVC", "fev_fvc_ratio"]

        # creates a correlation matrix
        corrmatrix = self.data.corr()

        # opens a file text file to output the results of the linear regression
        with open(f"slr_{metric}_independant.txt", "w") as f:
            # writes the correlation matrix to the output text file
            f.writelines(f"Correlation matrix for featues in dataset\n{corrmatrix}\n\n\n")

            # executes a linear regression for each of the features of interest
            for feature in features:
                # executes a ordinary least squares regression and stores the output in model
                model = smf.ols(f'{feature} ~ {metric}', self.data).fit()

                # writes the output of the model to the output text file
                f.writelines(f"{model.summary().as_text()}\n\n\n")

    def slr_copd_group(self, indep):
        # splits the data frame into 4 seperate dataframes based on COPD grouping
        data_mild = self.data[self.data["COPDSEVERITY"] == 0]
        data_mod = self.data[self.data["COPDSEVERITY"] == 1]
        data_sev = self.data[self.data["COPDSEVERITY"] == 2]
        data_vsev = self.data[self.data["COPDSEVERITY"] == 3]

        # creates 2 lists to containing 4 dataframes and each of the metrics being used in OLS regression
        datasets = [data_mild, data_mod, data_sev, data_vsev]
        metrics = ["MWT1Best", "FEV1", "FVC", "fev_fvc_ratio"]

        # opens an output file based on the independant variable inputted by the user
        with open(f"slr_{indep}_on_copd_by_group.txt", "w") as f:
            # a for loop to iterate through each of the datasets
            for df in datasets:
                # a for loop to iterate through each of the metrics
                for feature in metrics:
                    # fits the data to a OLS regression model using the current feature in the metric for loop and the independant variable selected by the user
                    model = smf.ols(f"{feature} ~ {indep}", df).fit()

                    # writes the summary of the OLS regression to the output file
                    f.writelines(f"COPD group = {copd_group_labeler(df, datasets)}, independant = {indep}\n")
                    f.writelines(model.summary().as_text())
                    f.writelines("\n\n\n")


# if __name__ == "main":
ps627 = COPD_project()
if ps627.mode == 0:
    ps627.data_splits()
    ps627.chisq_feat_sel()
    ps627.anova_feat_sel()
    ps627.reg_feat_sel()
    ps627.output_diagnostics()
if ps627.mode == 1:
    # ps627.copd_pred_logistic()
    ps627.slr("CAT")
    ps627.slr_copd_group("CAT")