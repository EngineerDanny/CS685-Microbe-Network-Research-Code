# combines the results for each parameter combination into a single result for analysis/plotting.
import pandas as pd
from glob import glob

# date_time = "2023-04-10_12:28"
# date_time = "2023-04-10_15:31"
# date_time = "2023-04-10_17:47"
date_time = "2023-05-29_12:40"
date_time = "2023-06-21_17:26"
date_time = "2023-06-21_19:01"
date_time = "2023-06-28_13:01"
date_time = "2023-07-19_16:19"

out_df_list = []
for out_csv in glob(f"/scratch/da2343/model_subsampling_{date_time}/results/*.csv"):
    out_df_list.append(pd.read_csv(out_csv))
error_df = pd.concat(out_df_list)

root_results_dir = "/projects/genomic-ml/da2343/ml_project_1/model_subsampling/results"
error_df.to_csv(f"{root_results_dir}/{date_time}_results.csv", index=False)