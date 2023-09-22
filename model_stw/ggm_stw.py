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

algorithm = "GGM"
index_of_pred_col = 0

dataset_path = dataset_dict[data_set_name]
n_splits = 3

# Import the csv file of the dataset
dataset_pd = pd.read_csv(dataset_path, header=0)
col_names = list(dataset_pd.columns)
# drop only one column per every iteration to form the input matrix
# make the column you removed the output
# print the size of the input matrix
output_vec = dataset_pd.iloc[:, index_of_pred_col].to_frame().to_numpy().ravel()
input_mat = dataset_pd.drop(dataset_pd.columns[index_of_pred_col], axis=1).to_numpy()

ggm_source_target_df_list = []


k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
for fold_id, indices in enumerate(k_fold.split(input_mat)):
    index_dict = dict(zip(["train", "test"], indices))
    set_data_dict = {}
    for set_name, index_vec in index_dict.items():
        set_data_dict[set_name] = {
            "X": input_mat[index_vec],
            "y": output_vec[index_vec]
        }
        
    X_train = set_data_dict["train"]["X"]
    y_train = set_data_dict["train"]["y"]
        
    source_target = get_glasso_source_target(
        X = X_train, 
        y = y_train, 
        index = index_of_pred_col,
    )
    source_target_df = pd.DataFrame(source_target, 
                                    columns=["source", "target", "weight"])
    source_target_df['fold_id'] = fold_id
    ggm_source_target_df_list.append(source_target_df)

final_source_target_df = pd.concat(ggm_source_target_df_list)
final_source_target_df = final_source_target_df.drop(['fold_id'], axis=1).reset_index(drop=True)
final_source_target_df = final_source_target_df.groupby(['source', 'target']).median()
final_source_target_df = final_source_target_df.reset_index()
print(final_source_target_df)

final_source_target_df.to_csv(f"/projects/genomic-ml/da2343/ml_project_1/model_stw/{data_set_name}_{algorithm}_source_target.csv", encoding='utf-8', index=False)
