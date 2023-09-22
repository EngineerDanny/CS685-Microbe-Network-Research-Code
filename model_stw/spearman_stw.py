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

# data_set_name = "necromass_bacteria_genus"
# data_set_name = "necromass_fungi_genus"
# data_set_name = "necromass_bacteria_fungi_genus"


data_set_name = "necromass_bacteria_conservative"
data_set_name = "necromass_fungi_conservative"
data_set_name = "necromass_bacteria_fungi_conservative"


algorithm = "Spearman"
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
    for set_name, index_vec in index_dict.items():
        set_data_dict[set_name] = {
            "X": X[index_vec],
            "y": y[index_vec]
        }
    my_learner = GridSearchCV(SpearmanRankRegressor(),
                        threshold_param_dict,
                        scoring='neg_mean_squared_error',
                        return_train_score=True)
    my_learner.fit(**set_data_dict["train"])
    X_train = set_data_dict["train"]["X"]
    y_train = set_data_dict["train"]["y"]
    X_train_ranked = ss.rankdata(X_train, axis=0)
    y_train_ranked = ss.rankdata(y_train)
    
    params_list = get_corr_hyper_params(X = X_train_ranked, 
                                        y = y_train_ranked, 
                                        cv_results = my_learner.cv_results_)
    mc_params_df = pd.DataFrame(params_list)
    # rename the column name threshold to reg_param
    mc_params_df.rename(columns={my_reg_param: 'reg_param'}, inplace=True)
    mc_params_df['subtrain_score'] = my_learner.cv_results_['mean_train_score'] * -1
    mc_params_df['validation_score'] = my_learner.cv_results_['mean_test_score'] * -1
    mc_params_df['index_of_pred_col'] = index_of_pred_col      
    return mc_params_df 

default_index_of_pred_col = 0
input_mat, output_vec = prep_data(dataset_pd, default_index_of_pred_col)
threshold_param_list = np.arange(0, 1, 0.1)
threshold_param_dict = [{'threshold': [threshold]}
                        for threshold in threshold_param_list]
my_reg_param = 'threshold'
source_target_df_list = []

index_of_pred_col_list = np.arange(0, dataset_pd.shape[1])
# index_of_pred_col_list = np.arange(0, 64)


k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
for fold_id, indices in enumerate(k_fold.split(input_mat)):
    corr_mc_df_list = []
    index_dict = dict(zip(["train", "test"], indices))
    pool = multiprocessing.Pool()
    corr_mc_df_list = pool.map(get_model_selection_params, index_of_pred_col_list)
    pool.close()
    pool.join()
    final_corr_df = pd.concat(corr_mc_df_list).reset_index( drop=True)
    final_corr_df = final_corr_df.groupby(['reg_param']).mean()
    final_corr_df = final_corr_df.drop(['index_of_pred_col'], axis=1)
    # find the best reg_param which is the index value of the min validation score
    best_reg_param = final_corr_df['validation_score'].idxmin()
    # use default_index_of_pred_col to get X_train and y_train
    X_train = input_mat[index_dict["train"]]
    y_train = output_vec[index_dict["train"]]
    
    X_train_ranked = ss.rankdata(X_train, axis=0)
    y_train_ranked = ss.rankdata(y_train)
    
    source_target = get_corr_source_target(
        X = X_train_ranked, 
        y = y_train_ranked, 
        index = default_index_of_pred_col,
        threshold = best_reg_param,
    )
    source_target_df = pd.DataFrame(source_target, 
                                    columns=["source", "target", "weight"])
    source_target_df['fold_id'] = fold_id
    source_target_df_list.append(source_target_df)
    
final_source_target_df = pd.concat(source_target_df_list)
# keeping source and target columns only, find the median of the weights and remove the fold_id column
final_source_target_df = final_source_target_df.drop(['fold_id'], axis=1).reset_index(drop=True)
final_source_target_df = final_source_target_df.groupby(['source', 'target']).median()
final_source_target_df = final_source_target_df.reset_index()

final_source_target_df.to_csv(f"/projects/genomic-ml/da2343/ml_project_1/model_stw/{data_set_name}_{algorithm}_source_target.csv", encoding='utf-8', index=False)
