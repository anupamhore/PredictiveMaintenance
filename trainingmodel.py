import pandas as pd
import numpy as np
from data_preprocessing import preprocessing
from data_preprocessing import clustering
from sklearn.model_selection import train_test_split
from best_model_finder import tuner
from file_ops import file_methods

class trainmodel:
    def __init__(self,path):
        self.fileobject = path
        self.df = pd.DataFrame()

    def trainingModel(self):
        self.df = pd.read_csv('Training/ai4i2020.csv')

        preprocessor = preprocessing.Preprocessor(self.df)

        #separate independant and dependant variables
        X,Y = preprocessor.separate_label_features('Air temperature [K]')

        #find missing value present and the columns
        is_null_present, cols_with_missing_values = preprocessor.is_null_present(X)

        #null value treatment
        if is_null_present:
            X = preprocessor.impute_missing_values(X, cols_with_missing_values)

        #columns to drop
        cols = ['UDI', 'Product ID']
        X = preprocessor.drop_variables(X,cols,1,True)

        #categorical dummy creation
        X = preprocessor.encode_categorical_columns(X)

        #handle outliers
        X = preprocessor.impute_outliers(X, ['Rotational speed [rpm]', 'Torque [Nm]'])

        #standard scaler
        data_scaled_X = preprocessor.scaleData(X)

        #VIF variables
        features_high_vif = preprocessor.find_variance_inflation_factor(data_scaled_X, X)


        #calculate multi-collinearity
        high_corr_columns = preprocessor.find_multi_collinieary_columns(X, 0.7)


        #remove the high VIF value columns
        X = preprocessor.drop_variables(X, features_high_vif, 1, True)

        #standard scaler
        data_scaled_X1 = preprocessor.scaleData(X)

        #VIF variables
        #features_high_vif1 = preprocessor.find_variance_inflation_factor(data_scaled_X1, X)


        pd.set_option('display.max_columns', None)

        #Applying clusteing methods
        kmeans = clustering.KMeansClustering(self.fileobject)
        number_of_clusters = kmeans.elbow_plot(data_scaled_X1)

        #Divide the data into clusters
        data_scaled_X2 = kmeans.create_clusters(data_scaled_X1, number_of_clusters)
        data_scaled_X2['Labels'] = Y

        list_of_unique_clusters = data_scaled_X2['Cluster'].unique()

        for cluster in list_of_unique_clusters:
            cluster_data = data_scaled_X2[data_scaled_X2['Cluster'] == cluster]

            #Prepare the dependant and independant variables
            cluster_features = cluster_data.drop(['Labels', 'Cluster'], axis=1)
            cluster_label = cluster_data['Labels']

            #split the data
            X_train, X_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=0.30, random_state=100)

            model_finder = tuner.Model_Finder(self.fileobject)

            #getting the best model for each of the clusters
            best_model_name, best_model = model_finder.get_best_model(X_train, y_train, X_test, y_test)

            #save the best model to the directory
            file_op = file_methods.File_Operation(self.fileobject)
            save_model = file_op.save_model(best_model, best_model_name+str(cluster))












