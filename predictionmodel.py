import pandas as pd
import numpy as np
from data_preprocessing import preprocessing
from data_preprocessing import clustering
from best_model_finder import tuner
from file_ops import file_methods

class predictmodel:

    def __init__(self, obj, path):
        self.predict_obj = obj
        self.fileobject = path

    def predictMass(self):

        try:
            self.df_new = pd.read_csv('Training/ai4i2020.csv')
            preprocessor = preprocessing.Preprocessor(self.df_new)

            # separate independant and dependant variables
            X, Y = preprocessor.separate_label_features('Air temperature [K]')

            is_null_present, cols_with_missing_values = preprocessor.is_null_present(X)

            # null value treatment
            if is_null_present:
                X = preprocessor.impute_missing_values(X, cols_with_missing_values)

            # columns to drop
            cols = ['UDI', 'Product ID']
            X = preprocessor.drop_variables(X, cols, 1, True)

            # categorical dummy creation
            X = preprocessor.encode_categorical_columns(X)

            # handle outliers
            X = preprocessor.impute_outliers(X, ['Rotational speed [rpm]', 'Torque [Nm]'])

            # standard scaler
            data_scaled_X = preprocessor.scaleData(X)

            # VIF variables
            features_high_vif = preprocessor.find_variance_inflation_factor(data_scaled_X, X)

            # calculate multi-collinearity
            high_corr_columns = preprocessor.find_multi_collinieary_columns(X, 0.7)

            # remove the multi-collinearity
            if len(list(high_corr_columns)) > 0:
                X = preprocessor.drop_variables(X, list(high_corr_columns), 1, True)

            # remove the high VIF value columns
            if len(features_high_vif) > 0:
                X = preprocessor.drop_variables(X, features_high_vif, 1, True)

            #standard scaler
            data_scaled_X1 = preprocessor.scaleData(X)

            file_loader = file_methods.File_Operation(self.fileobject)
            kmeans = file_loader.load_model('KMeans')

            X1 = pd.DataFrame(data_scaled_X1)

            clusters = kmeans.predict(X1)
            X1['clusters'] = clusters
            clusters = X1['clusters'].unique()

            for i in clusters:
                cluster_data = X1[X1['clusters'] == i]
                cluster_data = cluster_data.drop(['clusters'], axis=1)
                model_name = file_loader.find_correct_model_file(i)
                model = file_loader.load_model(model_name)
                result = model.predict(cluster_data)


            result_df = pd.DataFrame(list(zip(result)), columns=['Predictions'])
            final = pd.concat([X1,Y, result_df], axis=1)
            path = "Prediction_Output_File/Predictions.csv"
            final.to_csv("Prediction_Output_File/Predictions.csv", header=True, mode='a+')

        except Exception as e:
            raise Exception(e)

    def predict(self):

        try:
            pd.set_option('display.max_columns', None)
            self.df_new = pd.DataFrame(self.predict_obj, index=[0])
            self.df_new.columns = ['Type', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
                              'Tool wear [min]', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

            print(self.df_new)
            preprocessor = preprocessing.Preprocessor(self.df_new)

            # find missing value present and the columns
            is_null_present, cols_with_missing_values = preprocessor.is_null_present(self.df_new)

            # null value treatment
            if is_null_present:
                self.df_new = preprocessor.impute_missing_values(self.df_new, cols_with_missing_values)

            # categorical dummy creation
            self.df_new = preprocessor.encode_categorical_columns(self.df_new)


            # handle outliers
            self.df_new = preprocessor.impute_outliers(self.df_new, ['Rotational speed [rpm]', 'Torque [Nm]'])

            # standard scaler
            data_scaled_X = preprocessor.scaleData(self.df_new)

            # VIF variables
            features_high_vif = preprocessor.find_variance_inflation_factor(data_scaled_X, self.df_new)

            # calculate multi-collinearity
            high_corr_columns = preprocessor.find_multi_collinieary_columns(self.df_new, 0.7)

            # remove the multi-collinearity
            if len(list(high_corr_columns)) > 0:
                self.df_new = preprocessor.drop_variables(self.df_new,list(high_corr_columns), 1, True)

            # remove the high VIF value columns
            if len(features_high_vif) > 0:
                self.df_new = preprocessor.drop_variables(self.df_new, features_high_vif, 1, True)

            #standard scaler
            data_scaled_X1 = preprocessor.scaleData(self.df_new)

            file_loader = file_methods.File_Operation(self.fileobject)
            kmeans = file_loader.load_model('KMeans')

            X = pd.DataFrame(data_scaled_X1)

            clusters = kmeans.predict(X)
            X['clusters'] = clusters
            clusters = X['clusters'].unique()
            print(X)
            for i in clusters:
                cluster_data = X[X['clusters'] == i]
                cluster_data = cluster_data.drop(['clusters'], axis=1)
                model_name = file_loader.find_correct_model_file(i)
                model = file_loader.load_model(model_name)
                result = model.predict(cluster_data)
                print('Cluster:{} {}'.format(i, result[0]))
                return result[0]
        except Exception as e:
            raise Exception(e)


