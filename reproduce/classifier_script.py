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
classifier_dict = {
    "FeaturelessClassifier": DummyClassifier(strategy="most_frequent"),
    "LogisticRegressionCV": LogisticRegressionCV(random_state=1),
}

test_err_list = []
pred_col_name = df.columns[index_of_pred_col]

# two output vectors
# one will just be the output of the regression
output_vec_for_reg = df.iloc[:, index_of_pred_col].to_numpy().ravel()
# the other will be the output of the binary classifier
# when the elements in output_vec_for_reg is greater than 0 then the corresponding element in output_vec_classifier will be 1 otherwise it will be 0
output_vec_for_class = np.where(output_vec_for_reg > 0, 1, 0)
input_mat = df.drop(pred_col_name, axis=1).to_numpy()


k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
for fold_id, indices in enumerate(k_fold.split(input_mat)):
    index_dict = dict(zip(["train", "test"], indices))
    set_data_dict = {}
    for set_name, index_vec in index_dict.items():
        set_data_dict[set_name] = {
            "X": input_mat[index_vec],
            "y": output_vec_for_class[index_vec],
        }
    # Fit the learner to the training data
    # Predict the test data
    # Calculate the test error
    for learner_name, learner in classifier_dict.items():
        # check if y_train contains only one class
        if len(np.unique(set_data_dict["train"]["y"])) == 1:
            # predict all test data as the class that is present in y_train
            pred_y = np.full(set_data_dict["test"]["y"].shape, 
                             np.unique(set_data_dict["train"]["y"])[0])
        else:
            learner.fit(**set_data_dict["train"])
            pred_y = learner.predict(set_data_dict["test"]["X"])
        
        actual_y = set_data_dict["test"]["y"]
        fpr, tpr, _ = roc_curve(actual_y, pred_y)
        
        test_err_list.append(
            pd.DataFrame(
                {
                    "Test Accuracy": accuracy_score(actual_y, pred_y),
                    "FPR": np.array2string(fpr),
                    "TPR": np.array2string(tpr),
                    "FoldID": fold_id,
                    "Dataset": data_set_name,
                    "Index of Predicted Column": index_of_pred_col,
                    "Predicted Column Name": pred_col_name,
                    "Algorithm": learner_name
                },
                index=[0],
            )
        )
test_err_df = pd.concat(test_err_list)
# Save dataframe as a csv to output directory
out_file = f"results/{param_row}.csv"
test_err_df.to_csv(out_file, encoding="utf-8", index=False)
print("Done!!")