import pandas as pd
import numpy as np
from datetime import date
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import os


if len(sys.argv) != 3:
    sys.exit("Usage: python my_pearson.py col_start_index col_end_index")

try:
    start_col_index = int(sys.argv[1])
    end_col_index = int(sys.argv[2])
except:
    print("Please provide the fold index and the column range as arguments")
    sys.exit("Usage: python my_pearson.py col_start_index col_end_index")


# Name some string contants
out_dir = "/scratch/da2343/cs685fall22/data"
out_file = out_dir + \
    f"/pearson_model_multi_cols_{start_col_index}_{end_col_index}_{str(date.today())}_results.csv"
amgut_data_path = "./scaled_amgut_data_with_fold_ids_2022-10-28.csv"

# Import the csv file of the amgut1_data
amgut_data = pd.read_csv(amgut_data_path, header=0, index_col=0)
amgut_data = amgut_data.iloc[:, :-1]

data_dict = {}
n_row = amgut_data.shape[0]

data_dict["amgut_0"] = amgut_data
# should be in the range of 20s
for index in range(1, n_row-20, 10):
    sub_data_df = amgut_data.iloc[:-index, :]
    data_dict[f"amgut_{index}"] = sub_data_df

test_acc_df_list = []
for data_set_name, sub_data_df in data_dict.items():
    n_col = len(sub_data_df.columns)
    cor_df = pd.DataFrame(columns=['taxa1', 'taxa2', 'cor'])

    # Get cor_df from file system
    # cor_df = pd.read_csv('./cor_df.csv')
    # Filter it by the data_set_name column
    # cor_df = cor_df[cor_df['dataset']==data_set_name]

    for i in range(n_col):
        for j in range(n_col):
            if i != j:
                taxa1_col = sub_data_df.iloc[:, i]
                taxa2_col = sub_data_df.iloc[:, j]
                calc_cor = taxa1_col.corr(taxa2_col, method='pearson')
                abs_calc_cor = abs(calc_cor)

                # Calculate the standard deviation of taxa1 and taxa2 columns respectively
                sd_taxa1 = np.std(taxa1_col)
                sd_taxa2 = np.std(taxa2_col)

                mean_taxa1 = np.mean(sd_taxa1)
                mean_taxa2 = np.mean(sd_taxa2)

                grad = (calc_cor * sd_taxa2) / sd_taxa1
                intercept = (-grad * mean_taxa1) + mean_taxa2

                # Append the taxa1, taxa2, cor to the cor_df
                cor_df = pd.concat([cor_df,
                                    pd.DataFrame({'taxa1': sub_data_df.columns[i],
                                                    'taxa2': sub_data_df.columns[j],
                                                    'cor': calc_cor,
                                                    'grad': grad,
                                                    'intercept': intercept}, index=[0])
                                    ], axis=0, ignore_index=True)

    for pred_col_index in range(start_col_index, end_col_index):
        # Let X be all the columns/taxa except the dropped column/taxa of the sub_data dataframe
        X_features = sub_data_df.drop(
            amgut_data.columns[pred_col_index], axis=1)
        y_actual = sub_data_df.iloc[:, pred_col_index]
        # get the name of y
        y_name = sub_data_df.columns[pred_col_index]
        # Loop through all the columns in X
        y_pred_list = []
        for col_index in range(len(X_features.columns)):
            # get the name of the column
            x_name = X_features.columns[col_index]
            # get the column
            x_col = X_features.iloc[:, col_index]
            # find the combination of x_name and y_name in the cor_df
            # and get the gradient and intercept
            grad = cor_df.loc[(cor_df['taxa1'] == x_name) & (
                cor_df['taxa2'] == y_name), 'grad'].values[0]
            intercept = cor_df.loc[(cor_df['taxa1'] == x_name) & (
                cor_df['taxa2'] == y_name), 'intercept'].values[0]
            y_pred = np.array((grad * x_col) + intercept)
            y_pred_list.append(y_pred)
            y_pred_mean = np.mean(y_pred_list, axis=0)

            test_acc_dict = {
                "Mean Absolute Error": mean_absolute_error(y_actual.to_numpy(), y_pred_mean),
                "Mean Squared Error": mean_squared_error(y_actual.to_numpy(), y_pred_mean),
                "r2 Score": r2_score(y_actual.to_numpy(), y_pred_mean),
                "# of Samples": sub_data_df.shape[0],
                "Index of Predicted Column": pred_col_index,
                "Algorithm": "Pearson Correlation",
                "Data Set": "American Gut",
            }
            test_acc_df_list.append(pd.DataFrame(test_acc_dict, index=[0]))

test_acc_df = pd.concat(test_acc_df_list)


# Save dataframe as a csv to output directory
os.system("mkdir -p " + out_dir)
test_acc_df.to_csv(out_file, encoding='utf-8', index=False)
print("Done!!")
