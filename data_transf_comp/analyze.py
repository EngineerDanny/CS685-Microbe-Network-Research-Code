# combines the results for each parameter combination into a single result for analysis/plotting.
import pandas as pd
from glob import glob

# date_time = "2023-04-01_20:43"
# date_time = "2023-04-03_15:47"
# date_time = "2023-04-03_16:07"
# date_time = "2023-05-29_09:54"
# date_time = "2023-05-29_10:20"
date_time = "2023-05-29_14:34"

out_df_list = []
for out_csv in glob(f"/scratch/da2343/data_transf_comp_{date_time}/results/*.csv"):
    out_df_list.append(pd.read_csv(out_csv))
error_df = pd.concat(out_df_list)
error_df.to_csv(f"/home/da2343/cs685_fall22/data_transf_comp/results/{date_time}_results.csv", index=False)
