import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from kneed import KneeLocator
from file_ops import file_methods

#To show the graphs which are not called from main thread
plt.switch_backend('Agg')

class KMeansClustering:

    def __init__(self,path):
        self.fileobject = path

    def elbow_plot(self, data):

        wcss = []
        try:
            for i in range(1,11):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                kmeans.fit(data)
                wcss.append(kmeans.inertia_)
            plt.plot(range(1, 11), wcss)
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            plt.savefig('preprocessing_data/K-Means_Elbow.PNG')

            #finding the value of the optimum cluster programatically
            self.kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
            return self.kn.knee

        except Exception as e:
            raise Exception(e)

    def create_clusters(self, data, number_of_clusters):

        self.data = data

        try:
            self.kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
            self.pred_y_kmeans = self.kmeans.fit_predict(self.data)

            self.file_op = file_methods.File_Operation(self.fileobject)
            self.save_model = self.file_op.save_model(self.kmeans, 'KMeans')

            self.df = pd.DataFrame(self.data)
            self.df['Cluster'] = self.pred_y_kmeans

            return self.df

        except Exception as e:
            raise Exception(e)