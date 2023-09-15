import sys
import os
import pandas as pd
import warnings
import numpy as np
from datetime import date
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.abspath("/projects/genomic-ml/da2343/ml_project_1/shared"))
from model_header import *
from constants import *

warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)

params_df = pd.read_csv("params.csv")

if len(sys.argv) == 2:
    prog_name, task_str = sys.argv
    param_row = int(task_str)
else:
    print("len(sys.argv)=%d so trying first param" % len(sys.argv))
    param_row = 0

param_dict = dict(params_df.iloc[param_row, :])
data_set_name = param_dict["Dataset"]
n_sub_samples = param_dict["# of Total Samples"]
index_of_pred_col = param_dict["Index of Prediction Col"]

dataset_path = dataset_dict[data_set_name]
n_splits = 3
# Import the csv file of the dataset
dataset_pd = pd.read_csv(dataset_path, header=0)


threshold_param_dict = [
    {"threshold": [threshold]}
    for threshold in np.concatenate(
        (np.linspace(0, 0.4, 5), np.linspace(0.41, 0.6, 21), np.arange(0.7, 1.01, 0.1))
    )
]
learner_dict = {
    "Featureless": Featureless(),
    "Spearman": SpearmanRankRegressor(),
    "Pearson": MyPearsonRegressor(),
    "LASSO": LassoCV(random_state=1),
    "GGM": GaussianGraphicalModel(),
}


test_err_list = []

n_sections = int(np.floor(dataset_pd.shape[0] / n_sub_samples))
shuffled_df = dataset_pd.sample(frac=1, random_state=1)
total_samples = n_sub_samples * n_sections
shuffled_df_updated = shuffled_df.iloc[:total_samples, :]
shuffled_arr = np.split(shuffled_df_updated, n_sections)


for ss_index, sub_section in enumerate(shuffled_arr):
    # drop only one column per every iteration to form the input matrix
    # make the column you removed the output
    output_vec = sub_section.iloc[:, index_of_pred_col].to_numpy().ravel()
    input_mat = sub_section.drop(
        sub_section.columns[index_of_pred_col], axis=1
    ).to_numpy()

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
            mse = mean_squared_error(actual_y, pred_y)
            # r2_coef = r2_score(actual_y, pred_y)

            test_err_list.append(
                pd.DataFrame(
                    {
                        "Mean Squared Error": mse,
                        # "Root Mean Squared Error": np.sqrt(mse),
                        # "R Squared": pearsonr(actual_y, pred_y)[0] ** 2,
                        # "R2 Score": r2_coef,
                        "FoldID": fold_id,
                        "# of Total Samples": n_sub_samples,
                        "Index of Subsample": ss_index,
                        "Dataset": data_set_name,
                        "Index of Predicted Column": index_of_pred_col,
                        "Algorithm": learner_name,
                    },
                    index=[0],
                )
            )
main_test_err_df = pd.concat(test_err_list)

# Save dataframe as a csv to output directory
out_file = f"results/{param_row}.csv"
main_test_err_df.to_csv(out_file, encoding="utf-8", index=False)
print("Done!!")
