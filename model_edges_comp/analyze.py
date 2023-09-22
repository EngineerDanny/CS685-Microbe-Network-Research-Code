# combines the results for each parameter combination into a single result for analysis/plotting.
import pandas as pd
import os
from glob import glob


date_time = "2023-06-02_09:14" # amgut1
date_time = "2023-06-02_18:38" # amgut1
date_time = "2023-06-02_19:12" # amgut1
date_time = "2023-06-05_13:50"

# crohns
date_time = "2023-09-13_18:18"
date_time = "2023-09-13_20:38"

#ioral
# date_time = "2023-09-13_18:53"
date_time = "2023-09-13_21:04"
date_time = "2023-09-21_19:47"


project_dir = "/projects/genomic-ml/da2343/ml_project_1"
dir_name = "model_edges_comp"

pearson_df_list = []
for out_csv in glob(f"/scratch/da2343/{dir_name}_{date_time}/pearson_corr/*.csv"):
    pearson_df_list.append(pd.read_csv(out_csv))
output_dir = f"{project_dir}/{dir_name}/pearson_corr"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
pd.concat(pearson_df_list).to_csv(f"{output_dir}/pearson_corr_{date_time}.csv", index=False)

lasso_df_list = []
for out_csv in glob(f"/scratch/da2343/{dir_name}_{date_time}/lasso_coef/*.csv"):
    lasso_df_list.append(pd.read_csv(out_csv))
output_dir = f"{project_dir}/{dir_name}/lasso_coef"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
pd.concat(lasso_df_list).to_csv(f"{output_dir}/lasso_coef_{date_time}.csv", index=False)

source_target_df_list = []
for out_csv in glob(f"/scratch/da2343/{dir_name}_{date_time}/source_target/*.csv"):
    source_target_df_list.append(pd.read_csv(out_csv))
output_dir = f"{project_dir}/{dir_name}/source_target"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
pd.concat(source_target_df_list).to_csv(f"{output_dir}/source_target_{date_time}.csv", index=False)