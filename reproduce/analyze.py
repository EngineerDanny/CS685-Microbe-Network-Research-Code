# combines the results for each parameter combination into a single result for analysis/plotting.
import pandas as pd
from glob import glob

time = "2024-01-24_20:40"
time = "2024-01-30_13:25"
time = "2024-01-30_15:14"
time = "2024-01-30_15:31" # classifier
time = "2024-01-31_11:30" # classifier_reg
time = "2024-02-01_09:06" # classifier
time = "2024-02-01_09:27" # classifier
time = "2024-02-02_11:33" # classifier
time = "2024-02-04_19:53" # classifier_reg
time = "2024-02-04_22:36" # classifier_reg

time = "2024-02-05_10:50" # reg
time = "2024-02-05_12:00" # reg
time = "2024-02-05_22:08" # reg
time = "2024-02-06_11:56" # classifier_reg
time = "2024-02-06_14:07" # classifier_reg
time = "2024-02-06_14:55" # classifier_reg
time = "2024-02-06_15:08" # classifier_reg
time = "2024-02-06_15:45" # classifier_reg

time = "2024-02-06_21:47" # reg
time = "2024-02-09_11:36" # classifier
time = "2024-02-09_12:57" # classifier_reg

time = "2024-02-14_18:33" # classifier_reg
time = "2024-02-14_18:48" # classifier_reg
time = "2024-02-28_09:51" # classifier_reg

out_df_list = []
for out_csv in glob(f"/scratch/da2343/algo_comparison_{time}/results/*.csv"):
    out_df_list.append(pd.read_csv(out_csv))
error_df = pd.concat(out_df_list)

root_results_dir = "/projects/genomic-ml/da2343/ml_project_1/reproduce/results"
error_df.to_csv(f"{root_results_dir}/{time}_results.csv", index=False)

print('done')
