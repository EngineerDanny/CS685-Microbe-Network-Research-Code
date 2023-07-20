# combines the results for each parameter combination into a single result for analysis/plotting.
import pandas as pd
from glob import glob

# date_time = "2023-04-09_15:31"
# date_time = "2023-04-09_16:40"
date_time = "2023-04-22_14:53"
date_time = "2023-05-31_13:06"
date_time = "2023-05-31_17:24" # amgut1
date_time = "2023-06-01_14:32" # amgut2
date_time = "2023-06-01_15:06" # crohns
date_time = "2023-06-01_15:12" # ioral
date_time = "2023-06-21_20:44" # necromass_bacteria
date_time = "2023-07-06_14:38" # necromass_bacteria
date_time = "2023-07-07_13:52" # necromass_bacteria

model_comp_dir = "/projects/genomic-ml/da2343/ml_project_1/model_complexity"

# pearson_df_list = []
# for out_csv in glob(f"/scratch/da2343/model_complexity_{date_time}/pearson_corr/*.csv"):
#     pearson_df_list.append(pd.read_csv(out_csv))
# pd.concat(pearson_df_list).to_csv(f"{model_comp_dir}/pearson_corr/pearson_corr_{date_time}.csv", index=False)

# lasso_df_list = []
# for out_csv in glob(f"/scratch/da2343/model_complexity_{date_time}/lasso_coef/*.csv"):
#     lasso_df_list.append(pd.read_csv(out_csv))
# pd.concat(lasso_df_list).to_csv(f"{model_comp_dir}/lasso_coef/lasso_coef_{date_time}.csv", index=False)


# source_target_df_list = []
# algo = "pearson"
# for out_csv in glob(f"/scratch/da2343/model_complexity_{date_time}/{algo}_source_target/*.csv"):
#     source_target_df_list.append(pd.read_csv(out_csv))
# pd.concat(source_target_df_list).to_csv(f"{model_comp_dir}/source_target/{algo}_source_target_{date_time}.csv", index=False)
