# Import the modules
import os
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

date_time = "2023-09-14_20:09" # ioral
date_time = "2023-09-15_13:40" # crohns

date_time = "2023-10-23_11:59" # necromass_bacteria_fungi

model_comp_dir = "/projects/genomic-ml/da2343/ml_project_1/model_complexity"

# Create a list of subdirectories
subdirs = ["corr_mc_df", "lasso_coef", "pearson_stw", "spearman_stw", "ggm_stw"]

# Loop over the subdirectories
for subdir in subdirs:
    # Create the subdirectory if it does not exist
    os.makedirs(f"{model_comp_dir}/{subdir}", exist_ok=True)
    
    # Initialize an empty list to store the data frames
    df_list = []
    
    # Loop over the csv files in the subdirectory
    for out_csv in glob(f"/scratch/da2343/model_complexity_{date_time}/{subdir}/*.csv"):
        # Read the csv file and append it to the list
        df_list.append(pd.read_csv(out_csv))
    
    # Concatenate the data frames and save them as a single csv file
    pd.concat(df_list).to_csv(f"{model_comp_dir}/{subdir}/{date_time}.csv", index=False)
