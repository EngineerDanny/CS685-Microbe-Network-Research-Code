import sys
import os
import time
import pandas as pd
import warnings
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import *
sys.path.append(os.path.abspath("/projects/genomic-ml/da2343/ml_project_1/shared"))
from model_header import *
from constants import *

warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)



params_df = pd.read_csv("params.csv")
if len(sys.argv) == 2:
    prog_name, task_str = sys.argv
    param_row = int(task_str)
else:
    print("len(sys.argv)=%d so trying first param" % len(sys.argv))
    param_row = 1007
    
param_dict = dict(params_df.iloc[param_row, :])
data_set_name = param_dict["Dataset"]
index_of_pred_col = param_dict["Index of Prediction Col"]


dataset_path = dataset_dict[data_set_name]
n_splits = 3

print("got here 1")
# Import the csv file of the dataset
dataset_pd = pd.read_csv(dataset_path, header=0)
col_names = list(dataset_pd.columns)
# drop only one column per every iteration to form the input matrix
# make the column you removed the output
# print the size of the input matrix
output_vec = dataset_pd.iloc[:, index_of_pred_col].to_frame().to_numpy().ravel()
input_mat = dataset_pd.drop( dataset_pd.columns[index_of_pred_col], axis=1).to_numpy()


# threshold_param_list = np.concatenate(
#     (np.linspace(0, 0.2, 125), np.linspace(0.21, 0.4, 21), np.arange(0.5, 1.01, 0.1)))
# threshold_param_list = np.concatenate((np.linspace(0, 0.4, 5), 
#      np.linspace(0.41, 0.6, 21), np.arange(0.7, 1.01, 0.1)))
threshold_param_list = np.arange(0, 0.31, 0.1)
threshold_param_dict = [{'threshold': [threshold]}
                        for threshold in threshold_param_list]

# alpha_param_list = [10 ** x for x in np.concatenate((np.arange(-7, -2.5, 1), 
#                                        np.linspace(-2.5, -0.01, 50), 
#                                        np.arange(0, 1, 0.5)))]
alpha_param_list = [10 ** x for x in range(-10, 2)]
alpha_param_dict = [{'alpha': [alpha]} for alpha in alpha_param_list]

print("got here 2")

my_algorithm_list = [

    {
        'learner': GridSearchCV(SpearmanRankRegressor(),
                                threshold_param_dict,
                                scoring='neg_mean_absolute_error',
                                return_train_score=True,
                                cv=3,
                                ),
        'reg_param': 'threshold',
        'name': 'Spearman',
    },
    # },
]

print("got here 3")

lasso_coef_df_list = []
pearson_mc_df_list = []
source_target_df_list = []

pearson_source_target_df_list = []
spearman_source_target_df_list = []
ggm_source_target_df_list = []


start = time.time()

for i in range(1_000):
    k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    for fold_id, indices in enumerate(k_fold.split(input_mat)):
        print("got here 4")
        index_dict = dict(zip(["train", "test"], indices))
        set_data_dict = {}
        for set_name, index_vec in index_dict.items():
            set_data_dict[set_name] = {
                "X": input_mat[index_vec],
                "y": output_vec[index_vec]
            }

        for algorithm_dict in my_algorithm_list:
            my_learner = algorithm_dict["learner"]
            my_reg_param = algorithm_dict["reg_param"]
            my_algo_name = algorithm_dict["name"]
            print("got here 5")

            my_learner.fit(**set_data_dict["train"])
            X_train = set_data_dict["train"]["X"]
            y_train = set_data_dict["train"]["y"]
            X_train_ranked = ss.rankdata(X_train, axis=0)
            y_train_ranked = ss.rankdata(y_train)
            print("got here 6")
            
            if my_algo_name in ['Pearson', 'Spearman']:
                params_list = get_corr_hyper_params(X = X_train if my_algo_name == 'Pearson' else X_train_ranked, 
                                                        y = y_train if my_algo_name == 'Pearson' else y_train_ranked, 
                                                        cv_results = my_learner.cv_results_)
                pearson_mc_df = pd.DataFrame(params_list)
                # rename the column name threshold to reg_param
                pearson_mc_df.rename(columns={my_reg_param: 'reg_param'}, inplace=True)
                pearson_mc_df['subtrain_score'] = my_learner.cv_results_['mean_train_score'] * -1
                pearson_mc_df['validation_score'] = my_learner.cv_results_['mean_test_score'] * -1
                pearson_mc_df['algorithm'] = my_algo_name
                pearson_mc_df['data_set_name'] = data_set_name
                pearson_mc_df['fold_id'] = fold_id
                pearson_mc_df['index_of_pred_col'] = index_of_pred_col
                pearson_mc_df_list.append(pearson_mc_df)
                # Create the source-target dataframe
                best_reg_param = pearson_mc_df.loc[pearson_mc_df['validation_score'].idxmin(), 'reg_param']
                print("got here 7")
                print("best_reg_param: ", best_reg_param)
                
                source_target = get_corr_source_target(
                X = X_train if my_algo_name == 'Pearson' else X_train_ranked, 
                y = y_train if my_algo_name == 'Pearson' else y_train_ranked, 
                index = index_of_pred_col,
                threshold = best_reg_param,
                )
                source_target_df = pd.DataFrame(source_target, 
                                                columns=["source", "target", "weight"])
                source_target_df['algorithm'] = my_algo_name
                source_target_df['data_set_name'] = data_set_name
                source_target_df['fold_id'] = fold_id
                source_target_df['index_of_pred_col'] = index_of_pred_col
                source_target_df['threshold'] = best_reg_param
                
                if my_algo_name == 'Pearson':
                    pearson_source_target_df_list.append(source_target_df)
                else:
                    spearman_source_target_df_list.append(source_target_df)
                
                
            if my_algo_name == 'GGM' and index_of_pred_col == 0:
                source_target = get_glasso_source_target(
                    X = set_data_dict["train"]["X"], 
                    y = set_data_dict["train"]["y"], 
                    index = index_of_pred_col,
                )
                source_target_df = pd.DataFrame(source_target, 
                                                columns=["source", "target", "weight"])
                source_target_df['algorithm'] = my_algo_name
                source_target_df['data_set_name'] = data_set_name
                source_target_df['fold_id'] = fold_id
                source_target_df['index_of_pred_col'] = index_of_pred_col
                source_target_df['threshold'] = None
                ggm_source_target_df_list.append(source_target_df)

            if my_algo_name == 'LASSO':
                hyperparam_list = my_learner.cv_results_['params']
                mean_train_score_list = my_learner.cv_results_['mean_train_score'] * -1
                mean_test_score_list = my_learner.cv_results_['mean_test_score'] * -1
                
                lasso_mc_df_list = []
                estimator = Lasso()
                for hyperparam in hyperparam_list:
                    estimator.set_params(**hyperparam)
                    estimator.fit(**set_data_dict["train"])
                    coef_list = estimator.coef_.tolist()
                    coef_list.insert(index_of_pred_col, None)
                    coef_array = np.array(coef_list)
                    score_index = hyperparam_list.index(hyperparam)
                    lasso_mc_df_list.append(
                        {
                            'subtrain_score': mean_train_score_list[score_index],
                            'validation_score': mean_test_score_list[score_index],
                            'reg_param': hyperparam['alpha'],
                            'algorithm': my_algo_name,
                            'data_set_name': data_set_name,
                            'fold_id': fold_id,
                            'index_of_pred_col': index_of_pred_col,
                            'coefs': coef_array,
                        }
                    ) 
                lasso_coef_df_list.append(pd.DataFrame(lasso_mc_df_list))

    # final_pearson_corr_df = pd.concat(pearson_mc_df_list)
    # final_lasso_coef_df = pd.concat(lasso_coef_df_list)
    # final_pearson_source_target_df = pd.concat(pearson_source_target_df_list)
    final_spearman_source_target_df = pd.concat(spearman_source_target_df_list)
    # final_ggm_source_target_df = pd.concat(ggm_source_target_df_list)
    

    # if index_of_pred_col == 0:
    #     final_ggm_source_target_df = pd.concat(ggm_source_target_df_list)
        # final_ggm_source_target_df.to_csv(f"ggm_source_target/{param_row}.csv", encoding='utf-8', index=False)

    # Save dataframe as a csv to output directory
    # print(main_test_df)
    # main_test_df.to_csv(f"results/{param_row}.csv", encoding='utf-8', index=False)
    # final_lasso_coef_df.to_csv(f"lasso_coef/{param_row}.csv", encoding='utf-8', index=False)
    # final_pearson_corr_df.to_csv(f"pearson_corr/{param_row}.csv", encoding='utf-8', index=False)

    # final_pearson_source_target_df.to_csv(f"pearson_source_target/{param_row}.csv", encoding='utf-8', index=False)
    # final_spearman_source_target_df.to_csv(f"spearman_source_target/{param_row}.csv", encoding='utf-8', index=False)
end = time.time()
print(f"Runtime: {end - start} seconds")

print("Done!!")