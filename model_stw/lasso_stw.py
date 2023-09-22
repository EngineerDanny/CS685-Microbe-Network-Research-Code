import sys
import os
import pandas as pd
import warnings
import multiprocessing
import threading
import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import *
sys.path.append(os.path.abspath("/projects/genomic-ml/da2343/ml_project_1/shared"))
from model_header import *
from constants import *

warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)
    
# data_set_name = "necromass_bacteria"
# data_set_name = "necromass_fungi"
# data_set_name = "necromass_bacteria_fungi"
# data_set_name = "crohns"

# data_set_name = "necromass_bacteria_genus"
# data_set_name = "necromass_fungi_genus"
# data_set_name = "necromass_bacteria_fungi_genus"

data_set_name = "necromass_bacteria_conservative"
data_set_name = "necromass_fungi_conservative"
data_set_name = "necromass_bacteria_fungi_conservative"

algorithm = "LASSO"

dataset_path = dataset_dict[data_set_name]
n_splits = 3

# Import the csv file of the dataset
dataset_pd = pd.read_csv(dataset_path, header=0)
col_names = list(dataset_pd.columns)

def prep_data(df, index_of_pred_col):
    local_df = df.copy()
    y = local_df.iloc[:, index_of_pred_col].to_frame().to_numpy().ravel()
    X = local_df.drop(local_df.columns[index_of_pred_col], axis=1).to_numpy()
    return X, y

def get_model_selection_params(index_of_pred_col):
    X, y = prep_data(dataset_pd, index_of_pred_col)
    set_data_dict = {}
    # index_dict is a predefined variable
    for set_name, index_vec in index_dict.items():
        set_data_dict[set_name] = {
            "X": X[index_vec],
            "y": y[index_vec]
        }
        
    my_learner = GridSearchCV(Lasso(), 
                                alpha_param_dict,
                                scoring='neg_mean_squared_error', 
                                return_train_score=True)
    my_learner.fit(**set_data_dict["train"])
    hyperparam_list = my_learner.cv_results_['params']
    mean_train_score_list = my_learner.cv_results_['mean_train_score'] * -1
    mean_test_score_list = my_learner.cv_results_['mean_test_score'] * -1
            
    lasso_mc_df_list = []
    optim_coef_array = None
    min_validation_score = np.inf
    estimator = Lasso()
    for hyperparam in hyperparam_list:
        estimator.set_params(**hyperparam)
        estimator.fit(**set_data_dict["train"])
        coef_list = estimator.coef_.tolist()
        coef_list.insert(index_of_pred_col, None)
        coef_array = np.array(coef_list)
        score_index = hyperparam_list.index(hyperparam)
        if mean_test_score_list[score_index] < min_validation_score:
            min_validation_score = mean_test_score_list[score_index]
            optim_coef_array = coef_array
      
    return pd.DataFrame(
        {'coefs': [optim_coef_array],
         'index_of_pred_col': [index_of_pred_col],
         })
    
    

default_index_of_pred_col = 0
input_mat, output_vec = prep_data(dataset_pd, default_index_of_pred_col)

alpha_param_list = [10 ** x for x in range(-10, 2)]
alpha_param_dict = [{'alpha': [alpha]} for alpha in alpha_param_list]


source_target_df_list = []
index_of_pred_col_list = np.arange(0, dataset_pd.shape[1])

k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
for fold_id, indices in enumerate(k_fold.split(input_mat)):
    mc_df_list = []
    index_dict = dict(zip(["train", "test"], indices))
    # use multiprocessing to speed up the process
    pool = multiprocessing.Pool()
    mc_df_list = pool.map(get_model_selection_params, index_of_pred_col_list)
    pool.close()
    pool.join()
    final_coef_df = pd.concat(mc_df_list).reset_index(drop=True)
    # sort final_coef_df by the index_of_pred_col column in ascending order
    final_coef_df = final_coef_df.sort_values(by=['index_of_pred_col'])
    # drop the index_of_pred_col column
    final_coef_df = final_coef_df.drop(['index_of_pred_col'], axis=1)
    coefs_mat = np.array(final_coef_df['coefs'].to_list())
    coefs_mat = np.where(coefs_mat == None, np.nan, coefs_mat)
    # find the average of the upper and lower triangular matrix
    upper_tri = np.triu_indices(coefs_mat.shape[0], k=1)
    lower_tri = np.tril_indices(coefs_mat.shape[0], k=-1)
    coefs_mat_avg = (coefs_mat + coefs_mat.T) / 2
    coefs_mat[lower_tri] = coefs_mat_avg[lower_tri]
    coefs_mat[upper_tri] = np.nan
    np.fill_diagonal(coefs_mat, np.nan)
    coefs_mat = np.array(coefs_mat, dtype=float)
    source_target = np.argwhere(~np.isnan(coefs_mat))
    weights = coefs_mat[source_target[:, 0], source_target[:, 1]]
    source_target_result = [t for t in zip(source_target[:, 0], source_target[:, 1], weights) if np.abs(t[2]) > 0]
            
    # Create a dataframe with source, target and weight columns
    source_target_df = pd.DataFrame(source_target_result, 
                                            columns=["source", "target", "weight"])
    source_target_df['fold_id'] = fold_id
    source_target_df_list.append(source_target_df)

    
final_source_target_df = pd.concat(source_target_df_list)
# # keeping source and target columns only, find the median of the weights and remove the fold_id column
final_source_target_df = final_source_target_df.drop(['fold_id'], axis=1).reset_index(drop=True)
final_source_target_df = final_source_target_df.groupby(['source', 'target']).median()
final_source_target_df = final_source_target_df.reset_index()

final_source_target_df.to_csv(f"/projects/genomic-ml/da2343/ml_project_1/model_stw/{data_set_name}_{algorithm}_source_target.csv", encoding='utf-8', index=False)
# print(final_source_target_df)
