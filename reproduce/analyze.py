# combines the results for each parameter combination into a single result for analysis/plotting.
import pandas as pd
from glob import glob

date_time = "2024-01-24_20:40"
date_time = "2024-01-30_13:25"
date_time = "2024-01-30_15:14"
date_time = "2024-01-30_15:31" # classifier
date_time = "2024-01-31_11:30" # classifier_reg
date_time = "2024-02-01_09:06" # classifier
date_time = "2024-02-01_09:27" # classifier
date_time = "2024-02-02_11:33" # classifier
date_time = "2024-02-04_19:53" # classifier_reg
date_time = "2024-02-04_22:36" # classifier_reg

date_time = "2024-02-05_10:50" # reg


out_df_list = []
for out_csv in glob(f"/scratch/da2343/algo_comparison_{date_time}/results/*.csv"):
    out_df_list.append(pd.read_csv(out_csv))
error_df = pd.concat(out_df_list)

root_results_dir = "/projects/genomic-ml/da2343/ml_project_1/reproduce/results"
error_df.to_csv(f"{root_results_dir}/{date_time}_results.csv", index=False)

print('done')
