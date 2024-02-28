import sys
import os
import pandas as pd
import warnings
import numpy as np
from datetime import date
from sklearn.linear_model import *
from sklearn.model_selection import *
from sklearn.metrics import *
sys.path.append(os.path.abspath("/projects/genomic-ml/da2343/ml_project_1/shared"))
from model_header import *
from constants import *
from sklearn.dummy import *

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
index_of_pred_col = param_dict["Index of Prediction Col"]

dataset_path = dataset_dict[data_set_name]
n_splits = 3
# Import the csv file of the dataset
df = pd.read_csv(dataset_path, header=0)

classifier_reg_dict = {
    "Featureless": {
        "classifier": None,
        "regressor": Featureless(),
     },
    "LassoCV": {
        "classifier": None,
        "regressor": LassoCV(random_state=1),
    },
    "LogisticRegLassoCV": {
        "classifier": LogisticRegressionCV(),
        "regressor": LassoCV(random_state=1, cv=2),
    },
}    
    

test_err_list = []
pred_col_name = df.columns[index_of_pred_col]

# two output vectors
# one will just be the output of the regression
output_vec_for_reg = df.iloc[:, index_of_pred_col].to_numpy().ravel()
output_vec_for_class = np.where(output_vec_for_reg > 0, 1, 0)
input_mat = df.drop(pred_col_name, axis=1).to_numpy()


k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
for fold_id, indices in enumerate(k_fold.split(input_mat)):
    index_dict = dict(zip(["train", "test"], indices))
    set_data_dict = {}
    for set_name, index_vec in index_dict.items():
        set_data_dict[set_name] = {
            "X": input_mat[index_vec],
            "y_class": output_vec_for_class[index_vec],
            "y_reg": output_vec_for_reg[index_vec],
        }
    # Fit the learner to the training data
    # Predict the test data
    # Calculate the test error
    for learner_name, learner in classifier_reg_dict.items():
        # check if y_train contains only one class
        if len(np.unique(set_data_dict["train"]["y_class"])) == 1:
            # predict all test data as the class that is present in y_train
            pred_y = np.full(set_data_dict["test"]["y_class"].shape, 
                             np.unique(set_data_dict["train"]["y_class"])[0])
        else:
            classifier = learner["classifier"]
            if classifier != None:
                classifier.fit(set_data_dict["train"]["X"], set_data_dict["train"]["y_class"])
                classifier_pred_y = classifier.predict(set_data_dict["test"]["X"])
                # fit regressor only on rows where y_class is 1
                X_reg_train = set_data_dict["train"]["X"][set_data_dict["train"]["y_class"] == 1]
                y_reg_train = set_data_dict["train"]["y_reg"][set_data_dict["train"]["y_class"] == 1]
                
                try:
                    # check if y_reg_train is empty
                    if len(y_reg_train) != 0:
                        regressor = learner["regressor"]
                        regressor.fit(X_reg_train, y_reg_train)
                        regressor_pred_y = regressor.predict(set_data_dict["test"]["X"])
                        pred_y = np.where(classifier_pred_y == 0, 0, regressor_pred_y)
                    else:
                        pred_y = classifier_pred_y
                except Exception as e:
                    regressor = learner["regressor"]
                    regressor.fit(set_data_dict["train"]["X"], set_data_dict["train"]["y_reg"])
                    pred_y = regressor.predict(set_data_dict["test"]["X"])
            else:
                regressor = learner["regressor"]
                regressor.fit(set_data_dict["train"]["X"], set_data_dict["train"]["y_reg"])
                pred_y = regressor.predict(set_data_dict["test"]["X"])
            
        actual_y = set_data_dict["test"]["y_reg"]
        actual_pred_df = pd.DataFrame(
            {
                "Actual": actual_y,
                "Predicted": pred_y,
            }
        )
        actual_pred_df["FoldID"] = fold_id
        actual_pred_df["Dataset"] = data_set_name
        actual_pred_df["Index of Predicted Column"] = index_of_pred_col
        actual_pred_df["Predicted Column Name"] = pred_col_name
        actual_pred_df["Algorithm"] = learner_name
        test_err_list.append(actual_pred_df)
     
test_err_df = pd.concat(test_err_list)
# Save dataframe as a csv to output directory
out_file = f"results/{param_row}.csv"
test_err_df.to_csv(out_file, encoding="utf-8", index=False)
print("Done!!")