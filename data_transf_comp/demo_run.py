import pandas as pd
import numpy as np
from datetime import date
from sklearn.linear_model import LassoCV, BayesianRidge, LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sys
import os
from scipy.stats import pearsonr
from sklearn.preprocessing import *

sys.path.append(os.path.abspath("/projects/genomic-ml/da2343/ml_project_1/shared"))
from model_header import *
from constants import *


params_df = pd.read_csv("params.csv")

if len(sys.argv) == 2:
    prog_name, task_str = sys.argv
    param_row = int(task_str)
else:
    print("len(sys.argv)=%d so trying first param" % len(sys.argv))
    param_row = 0

param_dict = dict(params_df.iloc[param_row, :])
data_set_name = param_dict["Dataset"]
n_of_samples = param_dict["# of Samples"]
index_of_pred_col = param_dict["Index of Prediction Col"]

dataset_path = dataset_dict[data_set_name]
n_splits = 3

# Import the csv file of the dataset
dataset_pd = pd.read_csv(dataset_path, header=0)
sub_data_dict = {}
# drop only one column per every iteration to form the input matrix
# make the column you removed the output
# print the size of the input matrix
output_vec = dataset_pd.iloc[:, index_of_pred_col].to_frame()
input_mat = dataset_pd.drop(dataset_pd.columns[index_of_pred_col], axis=1)

input_mat_update = input_mat.iloc[:n_of_samples].to_numpy()
output_vec_update = output_vec.iloc[:n_of_samples].to_numpy().ravel()

data_tuple = (input_mat_update, output_vec_update, index_of_pred_col)

# Create a list of alphas for the LASSOCV to cross-validate against
threshold_param_list = np.concatenate(
    (np.linspace(0, 0.2, 125), np.linspace(0.21, 0.4, 21), np.arange(0.5, 1.01, 0.1))
)
threshold_param_dict = [
    {"threshold": [threshold]} for threshold in threshold_param_list
]

learner_dict = {
    "Featureless": Featureless(),
    # 'Pearson Correlation':  GridSearchCV(MyPearsonRegressor(), threshold_param_dict, cv=5, scoring='neg_mean_squared_error'),
    "Spearman": SpearmanRankRegressor(),
    "Pearson": MyPearsonRegressor(),
    "LassoCV": LassoCV(random_state=1),
    "GGM": GaussianGraphicalModel(),
}


test_err_list = []
pred_actual_list = []

(input_mat, output_vec, index_col) = data_tuple
k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
for fold_id, indices in enumerate(k_fold.split(input_mat)):
    index_dict = dict(zip(["train", "test"], indices))
    set_data_dict = {}
    for set_name, index_vec in index_dict.items():
        set_data_dict[set_name] = {
            "X": input_mat[index_vec],
            "y": output_vec[index_vec],
        }
    # Loop through the learners
    # Fit the learner to the training data
    # Predict the test data
    # Calculate the test error
    for learner_name, learner in learner_dict.items():
        learner.fit(**set_data_dict["train"])
        pred_y = learner.predict(set_data_dict["test"]["X"])
        actual_y = set_data_dict["test"]["y"]

        # Create a dataframe with the following columns:
        # Predicted labels, Actual labels, FoldID, # of Samples,
        # Dataset, Index of Predicted Column, Algorithm
        pred_actual_df = pd.DataFrame(
            {
                "Predicted Label": pred_y.tolist(),
                "Actual Label": actual_y.tolist(),
            }
        )
        pred_actual_df["FoldID"] = fold_id
        pred_actual_df["# of Train Samples"] = input_mat.shape[0]
        pred_actual_df["Dataset"] = data_set_name
        pred_actual_df["Index of Predicted Column"] = index_col
        pred_actual_df["Algorithm"] = learner_name
        pred_actual_list.append(pred_actual_df)

        mse = mean_squared_error(actual_y, pred_y)
        # r2_coef = r2_score(actual_y, pred_y)
        # calc pearson correlation between actual and predicted
        # pearson_coef = pearsonr(actual_y, pred_y)[0]

        test_err_list.append(
            pd.DataFrame(
                {
                    "Mean Squared Error": mse,
                    "Root Mean Squared Error": np.sqrt(mse),
                    # "R Squared": pearson_coef ** 2,
                    # "R2 Score": r2_coef,
                    "FoldID": fold_id,
                    "# of Train Samples": int(input_mat.shape[0] / n_splits),
                    "Dataset": data_set_name,
                    "Index of Predicted Column": index_col,
                    "Algorithm": learner_name,
                },
                index=[0],
            )
        )

main_test_err_df = pd.concat(test_err_list)
main_pred_actual_df = pd.concat(pred_actual_list)


# Save dataframe as a csv to output directory
out_file = f"results/{param_row}.csv"
# main_pred_actual_df.to_csv(out_file, encoding='utf-8', index=False)
main_test_err_df.to_csv(out_file, encoding="utf-8", index=False)
print("Done!!")
