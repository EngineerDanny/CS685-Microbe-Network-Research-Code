import pandas as pd
import numpy as np
from datetime import date
from sklearn.linear_model import LassoCV, BayesianRidge, LinearRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import *
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.stats import multivariate_normal
import scipy.stats as ss
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy import interpolate
from sklearn.pipeline import make_pipeline


class Featureless:
    def fit(self, X, y):
        self.mean = np.mean(y)

    def predict(self, X):
        test_features = X
        test_nrow, test_ncol = test_features.shape
        return np.repeat(self.mean, test_nrow)


class GaussianGraphicalModel:
    def fit(self, X, y):
        full_train_data = np.concatenate((y[:, None], X), axis=1)
        # ledoit_wolf_cov = ledoit_wolf(full_train_data)[0]
        # self.precision = np.linalg.inv(ledoit_wolf_cov)
        try:
            model = GraphicalLassoCV().fit(full_train_data)
            self.precision = model.precision_
        except:
            ledoit_wolf_cov = ledoit_wolf(full_train_data)[0]
            self.precision = np.linalg.inv(ledoit_wolf_cov)
        return self

    def predict(self, X):
        precision_first_val = self.precision[0, 0]
        constant_coef = (-1) / (2 * precision_first_val)
        mat_index = 0
        pre_y_list = []

        for col in X.T:
            I_1i = self.precision[0, mat_index + 1]
            I_i1 = self.precision[mat_index + 1, 0]
            I_1i_x = I_1i * col
            I_i1_x = I_i1 * col
            pre_y_list.append(np.add(I_1i_x, I_i1_x))
            mat_index += 1
        pre_y_sum = np.sum(np.array(pre_y_list), axis=0)
        pred_y = np.multiply(constant_coef, pre_y_sum)
        return pred_y

    def get_source_target(self, X, y, index):
        # concatenate X and y, insert y into the index column
        Xy = np.insert(X, index, y, axis=1)
        precision_matrix = GraphicalLassoCV().fit(Xy).precision_


class MyPearsonRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y):
        slope_list = []
        intercept_list = []
        for index_col in range(X.shape[1]):
            X_col = X[:, index_col]
            calc_slope, calc_intercept = self.find_model_params(X_col, y)
            slope_list.append(calc_slope)
            intercept_list.append(calc_intercept)
        # Find the mean of the gradients and intercepts
        self.slope_list = slope_list
        self.intercept_list = intercept_list
        return self

    def find_model_params(self, X_col, y_col):
        calc_cor = np.corrcoef(X_col, y_col)[0, 1]
        # If the correlation is greater than the threshold, then calculate the gradient and intercept
        if abs(calc_cor) > self.threshold:
            calc_slope = calc_cor * np.std(y_col) / np.std(X_col)
            calc_intercept = np.mean(y_col) - calc_slope * np.mean(X_col)
        else:
            calc_slope = None
            calc_intercept = None
        return calc_slope, calc_intercept

    def predict(self, X):
        pred_y_list = []
        for index_col in range(X.shape[1]):
            X_col = X[:, index_col]
            # use the average of the slope_list as the default slope
            filtered_slope_list = [x for x in self.slope_list if x is not None]
            mean_filtered_slope = (
                np.mean(filtered_slope_list) if len(filtered_slope_list) > 0 else 0
            )
            calc_slope = (
                mean_filtered_slope
                if self.slope_list[index_col] is None
                else self.slope_list[index_col]
            )

            filtered_intercept_list = [x for x in self.intercept_list if x is not None]
            mean_filtered_intercept = (
                np.mean(filtered_intercept_list)
                if len(filtered_intercept_list) > 0
                else 0
            )
            calc_intercept = (
                mean_filtered_intercept
                if self.intercept_list[index_col] is None
                else self.intercept_list[index_col]
            )

            calc_y = calc_slope * X_col + calc_intercept
            pred_y_list.append(calc_y)
        # Find the mean of the predicted y values
        pred_y = np.mean(pred_y_list, axis=0)
        return pred_y


class SpearmanRankRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.preprocessor1 = make_pipeline(
            MinMaxScaler(),
            StandardScaler(),
        )
        self.preprocessor2 = make_pipeline(
            MinMaxScaler(),
            StandardScaler(),
        )

    def fit(self, X, y):
        self.y_train = y
        # X_train_ranked_transf = ss.rankdata(X, axis=0)
        self.y_train_ranked_transf = self.preprocessor2.fit_transform(
            ss.rankdata(y).reshape(-1, 1)
        ).flatten()
        # self.y_train_ranked_transf = ss.rankdata(y)
        X_train_ranked_transf = self.preprocessor1.fit_transform(ss.rankdata(X, axis=0))
        # X_train_ranked_transf = PowerTransformer().fit_transform(ss.rankdata(X, axis=0))
        # self.y_train_ranked_transf = PowerTransformer().fit_transform(ss.rankdata(y))

        slope_list = []
        intercept_list = []

        for index_col in range(X_train_ranked_transf.shape[1]):
            X_col = X_train_ranked_transf[:, index_col]
            calc_slope, calc_intercept = self.find_model_params(
                X_col, self.y_train_ranked_transf
            )
            slope_list.append(calc_slope)
            intercept_list.append(calc_intercept)
        # Find the mean of the gradients and intercepts
        self.slope_list = slope_list
        self.intercept_list = intercept_list
        return self

    def find_model_params(self, X_col, y_col):
        calc_cor = np.corrcoef(X_col, y_col)[0, 1]
        # If the correlation is greater than the threshold, then calculate the gradient and intercept
        if abs(calc_cor) > self.threshold:
            calc_slope = calc_cor * np.std(y_col) / np.std(X_col)
            calc_intercept = np.mean(y_col) - calc_slope * np.mean(X_col)
        else:
            calc_slope = None
            calc_intercept = None
        return calc_slope, calc_intercept

    def predict(self, X):
        pred_y_list = []
        X_test_ranked_transf = self.preprocessor1.fit_transform(ss.rankdata(X, axis=0))
        # X_test_ranked_transf = ss.rankdata(X, axis=0)

        for index_col in range(X_test_ranked_transf.shape[1]):
            X_col = X_test_ranked_transf[:, index_col]
            # use the average of the slope_list as the default slope
            filtered_slope_list = [x for x in self.slope_list if x is not None]
            mean_filtered_slope = (
                np.mean(filtered_slope_list) if len(filtered_slope_list) > 0 else 0
            )
            calc_slope = (
                mean_filtered_slope
                if self.slope_list[index_col] is None
                else self.slope_list[index_col]
            )

            filtered_intercept_list = [x for x in self.intercept_list if x is not None]
            mean_filtered_intercept = (
                np.mean(filtered_intercept_list)
                if len(filtered_intercept_list) > 0
                else 0
            )
            calc_intercept = (
                mean_filtered_intercept
                if self.intercept_list[index_col] is None
                else self.intercept_list[index_col]
            )

            calc_y_ranked = calc_slope * X_col + calc_intercept

            # remove duplicate values from self.y_train_ranked_transf and use indexes to remove items from self.y_train
            y_train_ranked_transf_unique, sorted_indexes = np.unique(
                self.y_train_ranked_transf, return_index=True
            )
            y_train_unique = self.y_train[sorted_indexes]

            try:
                linear_interpolation = interpolate.interp1d(
                    y_train_ranked_transf_unique,
                    y_train_unique,
                    fill_value="extrapolate",
                )
                calc_y = linear_interpolation(calc_y_ranked)
                if np.isnan(calc_y).any():
                    calc_y = [np.mean(self.y_train)] * len(calc_y_ranked)
            except Exception as e:
                calc_y = [np.mean(self.y_train)] * len(calc_y_ranked)

            pred_y_list.append(calc_y)
        # Find the mean of the predicted y values
        pred_y = np.mean(np.array(pred_y_list), axis=0)
        return pred_y


class SpearmanRankRegressorTest(BaseEstimator, RegressorMixin):
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.preprocessor = MinMaxScaler()

    def fit(self, X, y):
        self.y_train = y
        self.y_train_ranked_transf = y
        X_train_ranked_transf = self.preprocessor.fit_transform(
            ss.rankdata(X, axis=0), ss.rankdata(y)
        )

        slope_list = []
        intercept_list = []

        for index_col in range(X_train_ranked_transf.shape[1]):
            X_col = X_train_ranked_transf[:, index_col]
            calc_slope, calc_intercept = self.find_model_params(
                X_col, self.y_train_ranked_transf
            )
            slope_list.append(calc_slope)
            intercept_list.append(calc_intercept)
        self.slope_list = slope_list
        self.intercept_list = intercept_list
        return self

    def find_model_params(self, X_col, y_col):
        calc_cor = np.corrcoef(X_col, y_col)[0, 1]
        if abs(calc_cor) > self.threshold:
            calc_slope = calc_cor * np.std(y_col) / np.std(X_col)
            calc_intercept = np.mean(y_col) - calc_slope * np.mean(X_col)
        else:
            calc_slope = None
            calc_intercept = None
        return calc_slope, calc_intercept

    def predict(self, X):
        pred_y_list = []
        X_test_ranked_transf = self.preprocessor.transform(ss.rankdata(X, axis=0))

        for index_col in range(X_test_ranked_transf.shape[1]):
            X_col = X_test_ranked_transf[:, index_col]
            filtered_slope_list = [x for x in self.slope_list if x is not None]
            mean_filtered_slope = (
                np.mean(filtered_slope_list) if len(filtered_slope_list) > 0 else 0
            )
            calc_slope = (
                mean_filtered_slope
                if self.slope_list[index_col] is None
                else self.slope_list[index_col]
            )

            filtered_intercept_list = [x for x in self.intercept_list if x is not None]
            mean_filtered_intercept = (
                np.mean(filtered_intercept_list)
                if len(filtered_intercept_list) > 0
                else 0
            )
            calc_intercept = (
                mean_filtered_intercept
                if self.intercept_list[index_col] is None
                else self.intercept_list[index_col]
            )

            calc_y_ranked = calc_slope * X_col + calc_intercept

            y_train_ranked_transf_unique, sorted_indexes = np.unique(
                self.y_train_ranked_transf, return_index=True
            )
            y_train_unique = self.y_train[sorted_indexes]

            linear_interpolation = interpolate.interp1d(
                y_train_ranked_transf_unique, y_train_unique, fill_value="extrapolate"
            )

            calc_y = linear_interpolation(calc_y_ranked)

            pred_y_list.append(calc_y)
        # Find the mean of the predicted y values
        pred_y = np.mean(np.array(pred_y_list), axis=0)
        return pred_y


### HELPER FUNCTIONS ###
# Returns the number of edges in the graph
# based on the threshold on the Pearson correlation matrix
def get_corr_hyper_params(X, y, cv_results):
    threshold_list = cv_results["params"]
    param_list = []
    # concatenate X and y
    Xy = np.concatenate((X, y.reshape(-1, 1)), axis=1)
    for threshold_dict in threshold_list:
        threshold = threshold_dict["threshold"]
        # calculate the Pearson correlation matrix
        corr_matrix = np.corrcoef(Xy, rowvar=False)
        np.fill_diagonal(corr_matrix, np.nan)
        corr_matrix = np.tril(corr_matrix)
        abs_corr_matrix = np.abs(corr_matrix)
        # get the number of edges
        no_of_edges = np.sum(abs_corr_matrix > threshold)
        # append the number of edges and the threshold to the list as dictionary
        param_list.append({"threshold": threshold, "edges": no_of_edges})
    return param_list


# Returns a list of source, target, weight tuples for optimum correlation matrix
def get_corr_source_target(X, y, index, threshold):
    # concatenate X and y, insert y into the index column
    Xy = np.insert(X, index, y, axis=1)
    # Xy = np.concatenate((X, y.reshape(-1, 1)), axis=1)
    # calculate the Pearson correlation matrix
    corr_matrix = np.corrcoef(Xy, rowvar=False)
    np.fill_diagonal(corr_matrix, np.nan)
    # get the indices of the upper triangle elements
    upper_tri = np.triu_indices(corr_matrix.shape[0], k=1)
    # assign np.nan to those elements
    corr_matrix[upper_tri] = np.nan
    abs_corr_matrix = np.abs(corr_matrix)
    # get the source and target nodes
    source_target = np.argwhere(abs_corr_matrix > threshold)
    # get the corresponding weights from the original correlation matrix
    weights = corr_matrix[source_target[:, 0], source_target[:, 1]]
    # create a list of source, target, weight tuples
    result = list(zip(source_target[:, 0], source_target[:, 1], weights))
    return result


# Returns a list of source, target, weight tuples for the precision matrix
def get_glasso_source_target(X, y, index):
    # concatenate X and y, insert y into the index column
    Xy = np.insert(X, index, y, axis=1)
    # get the precision matrix
    try:
        precision_matrix = GraphicalLassoCV(n_jobs=-1).fit(Xy).precision_
    except:
        ledoit_wolf_cov = ledoit_wolf(Xy)[0]
        precision_matrix = np.linalg.inv(ledoit_wolf_cov)
    # replace the diagonal with np.nan
    np.fill_diagonal(precision_matrix, np.nan)
    # get the indices of the upper and lower triangle elements
    upper_tri = np.triu_indices(precision_matrix.shape[0], k=1)
    lower_tri = np.tril_indices(precision_matrix.shape[0], k=-1)
    # calculate the average of the upper and lower triangle elements
    avg_matrix = (precision_matrix + precision_matrix.T) / 2
    # assign the average values to the lower triangle elements
    precision_matrix[lower_tri] = avg_matrix[lower_tri]
    # assign np.nan to the upper triangle elements
    precision_matrix[upper_tri] = np.nan
    # get the source and target nodes
    source_target = np.argwhere(~np.isnan(precision_matrix))
    # get the corresponding weights from the absolute precision matrix
    weights = precision_matrix[source_target[:, 0], source_target[:, 1]]
    # create a list of source, target, weight tuples with non-zero weights
    result = [
        t for t in zip(source_target[:, 0], source_target[:, 1], weights) if t[2] != 0
    ]
    return result
