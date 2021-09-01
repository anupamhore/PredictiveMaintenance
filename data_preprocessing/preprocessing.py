import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import  variance_inflation_factor

class Preprocessor:

    def __init__(self,df):
        self.df = df

    def separate_label_features(self,labelName):

        try:
            self.X = self.df.drop(labelName,axis=1)
            self.Y = self.df[labelName]
            return self.X,self.Y
        except Exception as e:
            raise Exception(e)

    def is_null_present(self, data):

        self.null_present = False
        self.columns_with_missing_values = []
        self.cols = data.columns

        try:
            self.null_counts = data.isna().sum()
            for i in range(len(self.null_counts)):
                if self.null_counts[i] > 0:
                    self.null_present = True
                    self.columns_with_missing_values.append(self.cols[i])

            return self.null_present, self.columns_with_missing_values

        except Exception as e:
            raise Exception(e)

    def impute_missing_values(self, data, cols_with_missing_values):

        self.data = data
        try:

            for col in cols_with_missing_values:
                self.data[col] = self.data[col].fillna(self.data[col].median())
            return self.data
        except Exception as e:
            raise Exception(e)

    def drop_variables(self, data, columns, axis, inplace):

        self.data = data
        try:
            self.data.drop(columns, axis=axis, inplace=inplace)
            return self.data
        except Exception as e:
            raise Exception(e)

    def encode_categorical_columns(self, data):

        self.data = data
        try:
            cat_features = [i for i in data.columns if data[i].dtypes == 'object']
            for feature in cat_features:
                if len(self.data[feature].unique()) > 1:

                    col_dummies = pd.get_dummies(self.data[feature], prefix=feature, drop_first=True)
                    self.data = pd.concat([self.data, col_dummies], axis=1)
                    self.data.drop(feature, axis=1, inplace=True)
                    return self.data
                else:
                    self.data[feature] = self.data[feature].map(lambda x: 1)
                    return self.data

        except Exception as e:
            raise Exception(e)

    def impute_outliers(self, data, cols):

        self.data = data

        try:
            for variable in cols:
                IQR = self.data[variable].quantile(0.75) - self.data[variable].quantile(0.25)
                lower_bridge = self.data[variable].quantile(0.25) - 1.5 * IQR
                upper_bridge = self.data[variable].quantile(0.75) + 1.5 * IQR
                self.data[variable].clip(lower=lower_bridge, inplace=True)
                self.data[variable].clip(upper=upper_bridge, inplace=True)
            return self.data

        except Exception as e:
            raise Exception(e)


    def scaleData(self, data):

        self.data = data
        try:
            scaler = StandardScaler()
            arr = scaler.fit_transform(self.data)
            return arr
        except Exception as e:
            raise Exception(e)

    def find_variance_inflation_factor(self, arr, X):

        self.arr = arr

        try:
            vif = pd.DataFrame()
            vif['VIF'] = [variance_inflation_factor(self.arr, i) for i in range(self.arr.shape[1])]
            vif['Features'] = X.columns
            series = vif[vif['VIF'] > 10]
            arr = list(series.Features)
            return arr

        except Exception as e:
            raise Exception(e)

    def find_multi_collinieary_columns(self, data, threshold):

        self.data = data

        try:
            col_corr = set()
            cor_matrix = self.data.corr()
            for i in range(len(cor_matrix.columns)):
                for j in range(i):
                    if cor_matrix.iloc[i, j] > threshold:
                        colName = cor_matrix.columns[i]
                        col_corr.add(colName)

            return col_corr

        except Exception as e:
            raise Exception(e)