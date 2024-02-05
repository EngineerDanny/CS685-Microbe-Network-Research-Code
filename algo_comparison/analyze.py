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
date_time = "2023-07-19_20:27"
date_time = "2023-07-20_16:36"
date_time = "2023-07-20_16:44"
date_time = "2023-08-03_18:13"
date_time = "2023-08-18_14:28"

# necromass genus level OTU data
date_time = "2023-09-15_18:01"
# necromass species level OTU data
date_time = "2023-09-15_20:03"

date_time = "2023-10-02_08:20"
date_time = "2023-10-02_08:33"
date_time = "2023-10-02_09:44"
date_time = "2023-10-02_11:20"
date_time = "2023-10-02_14:35"
date_time = "2023-10-02_18:42"
date_time = "2023-10-02_20:15"

date_time = "2023-10-09_21:20"
date_time = "2023-10-09_22:56"
date_time = "2023-10-10_09:23"
date_time = "2023-10-10_10:37"
date_time = "2023-10-12_22:17"
date_time = "2024-02-05_08:41"


out_df_list = []
for out_csv in glob(f"/scratch/da2343/algo_comparison_{date_time}/results/*.csv"):
    out_df_list.append(pd.read_csv(out_csv))
error_df = pd.concat(out_df_list)

root_results_dir = "/projects/genomic-ml/da2343/ml_project_1/algo_comparison/results"
error_df.to_csv(f"{root_results_dir}/{date_time}_results.csv", index=False)

print('done')
