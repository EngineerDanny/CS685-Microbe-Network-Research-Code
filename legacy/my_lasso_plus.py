import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error
import time
from datetime import date
import os

# Name some string contants
out_dir = "/scratch/da2343/cs685fall22/data"
out_file = out_dir + \
    f"/my_lasso_results.csv"
    
    
amgut_data_path = "./amgut1_data.csv"


amgut_data = pd.read_csv(amgut_data_path)
# remove first column
amgut_data = amgut_data.iloc[:, 1:]

data_dict = {}

(n_rows, n_cols) = amgut_data.shape

# drop only one column per every iteration to form the input matrix
# make the column you removed the output
# print the size of the input matrix
for index_col in range(n_cols):
    output_vec = amgut_data.iloc[:, index_col].to_frame()
    input_mat = amgut_data.drop(amgut_data.columns[index_col], axis=1)
    # Subset the data by decreasing the number of rows per iteration
    for index_row in range(1, n_rows-20, 20):
        input_mat_update = input_mat.iloc[:-index_row, :].to_numpy()
        input_mat_update_scaled = np.where(np.std(input_mat_update, axis=0) == 0, input_mat_update, (
            input_mat_update - np.mean(input_mat_update, axis=0)) / np.std(input_mat_update, axis=0))

        output_vec_update = output_vec.iloc[:-index_row, :].to_numpy().ravel()
        output_vec_update_scaled = np.where(np.std(output_vec_update, axis=0) == 0, output_vec_update, (
            output_vec_update - np.mean(output_vec_update, axis=0)) / np.std(output_vec_update, axis=0))

        data_dict[f"amgut_r_{index_row}_c_{index_col}"] = (
            input_mat_update_scaled, output_vec_update_scaled, index_col)


n_splits = 3
test_acc_df_list = []
for data_set, (input_mat, output_vec, pred_col) in data_dict.items():
    k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    mean_test_error = 0
    for fold_id, indices in enumerate(k_fold.split(input_mat)):
        index_dict = dict(zip(["train", "test"], indices))
        set_data_dict = {}
        for set_name, index_vec in index_dict.items():
            set_data_dict[set_name] = {
                "X": input_mat[index_vec],
                "y": output_vec[index_vec]
            }
        lasso_cv = LassoCV(cv=5, random_state=0,
                           max_iter=10000, n_jobs=-1)
        # time.sleep(0.1)
        lasso_cv.fit(**set_data_dict["train"])
        # time.sleep(0.1)
        test_data_x = set_data_dict["test"]['X']
        test_data_y = set_data_dict["test"]['y']
        pred_vec = lasso_cv.predict(test_data_x)
        mean_abs_error = mean_absolute_error(test_data_y, pred_vec)
        mean_test_error += mean_abs_error

    mean_test_error /= n_splits
    test_acc_dict = {
        "test_error": mean_test_error,
        "# of Samples": input_mat.shape[0],
        "Index of Predicted Column": pred_col,
        "Algorithm": "Pearson Correlation",
        "algorithm": "LassoCV"
    }
    test_acc_df_list.append(pd.DataFrame(test_acc_dict, index=[0]))

# For every # of sample index, find the average of the test error
# and plot the test error vs # of samples
test_acc_df = pd.concat(test_acc_df_list)
# Save dataframe as a csv to output directory
os.system("mkdir -p " + out_dir)
test_acc_df.to_csv(out_file, encoding='utf-8', index=False)
print("Done!!")
