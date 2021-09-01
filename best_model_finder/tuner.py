from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso,LassoCV, Ridge, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np

class Model_Finder:

    def __init__(self,file_object):
        self.file_object = file_object

    def get_best_params_for_linearRegression(self, x, y):

        try:
            self.linear_reg = LinearRegression()
            self.linear_reg.fit(x, y)
            return self.linear_reg

        except Exception as e:
            raise Exception(e)

    def get_best_params_for_lassoRegression(self, x, y):

        try:
            lasso_params = {'alpha': [0.02, 0.024, 0.025, 0.026, 0.03]}
            lasso_cv = LassoCV(alphas=lasso_params['alpha'], cv=10, max_iter=2000000, normalize=True)
            lasso_cv.fit(x, y)
            lasso = Lasso(alpha=lasso_cv.alpha_)
            lasso.fit(x, y)
            return lasso

        except Exception as e:
            raise Exception(e)

    def get_best_params_for_ridgeRegression(self, x, y):

        try:
            ridge_params = {'alpha': [200, 230, 250, 265, 270, 275, 290, 300, 500]}
            ridgeCV = RidgeCV(alphas=ridge_params['alpha'], cv = 10, normalize=True)
            ridgeCV.fit(x, y)

            ridge = Ridge(alpha=ridgeCV.alpha_)
            ridge.fit(x, y)
            return ridge

        except Exception as e:
            raise Exception(e)

    def get_best_params_for_elasticNetRegression(self, x, y):

        try:
            elasticNetCV = ElasticNetCV(l1_ratio= np.arange(0, 1, 0.01), alphas=[0, 0.5, 0.1, 0.01, 0.001],max_iter=2000000,normalize=True)
            elasticNetCV.fit(x, y)
            elasticnet = ElasticNet(alpha=elasticNetCV.alpha_,l1_ratio=elasticNetCV.l1_ratio_)
            elasticnet.fit(x, y)
            return elasticnet

        except Exception as e:
            raise Exception(e)


    def get_best_model(self, X_train, y_train, X_test, y_test):

        try:
            #Linear Regression
            self.linear_reg = self.get_best_params_for_linearRegression(X_train, y_train)
            self.linear_reg_score = self.linear_reg.score(X_test, y_test)

            #Lasso Regression
            self.lasso_reg = self.get_best_params_for_lassoRegression(X_train, y_train)
            self.lasso_reg_score = self.lasso_reg.score(X_test, y_test)

            #Ridge Regression
            self.ridge_reg = self.get_best_params_for_ridgeRegression(X_train, y_train)
            self.ridge_reg_score = self.ridge_reg.score(X_test, y_test)

            #ElasticNet Regression
            self.elastic_net_reg = self.get_best_params_for_elasticNetRegression(X_train, y_train)
            self.elastic_net_reg_score = self.elastic_net_reg.score(X_test, y_test)

            self.scoreList = [{"modelName":"Linear Regression", "modelscore":self.linear_reg_score, "model":self.linear_reg},
                              {"modelName":"Lasso", "modelscore":self.lasso_reg_score, "model":self.lasso_reg},
                              {"modelName":"Ridge", "modelscore":self.ridge_reg_score, "model":self.ridge_reg},
                              {"modelName":"Elastic Net", "modelscore":self.elastic_net_reg_score, "model":self.elastic_net_reg}]
            self.scoreList.sort(key=lambda x: x['modelscore'], reverse=True)
            modelObject = self.scoreList[0]
            return modelObject['modelName'], modelObject['model']

        except Exception as e:
            raise Exception(e)


