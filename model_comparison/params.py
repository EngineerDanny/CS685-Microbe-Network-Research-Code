from datetime import datetime
import pandas as pd
import numpy as np
import os
import shutil


def get_df_from_path(path):
    df = pd.read_csv(path, header=0)
    return df


dataset_list = [

    # {"dataset_name": "amgut1",
    #     "dataframe": get_df_from_path("/home/da2343/cs685_fall22/data/amgut1_data_power_transformed.csv")},

    # {"dataset_name": "amgut1_standard_scaled",
    #     "dataframe": get_df_from_path("/home/da2343/cs685_fall22/data/amgut1_data_standard_scaled.csv")},

    # {"dataset_name": "amgut1_log1_standard_scaled",
    #     "dataframe": get_df_from_path("/home/da2343/cs685_fall22/data/amgut1_data_log1_standard_scaled_transformed.csv")},
    
    
    
    {"dataset_name": "amgut2",
        "dataframe": get_df_from_path("/home/da2343/cs685_fall22/data/amgut2_data_power_transformed.csv")},

    {"dataset_name": "amgut2_standard_scaled",
        "dataframe": get_df_from_path("/home/da2343/cs685_fall22/data/amgut2_data_standard_scaled.csv")},

    {"dataset_name": "amgut2_log1_standard_scaled",
        "dataframe": get_df_from_path("/home/da2343/cs685_fall22/data/amgut2_data_log1_standard_scaled_transformed.csv")},



    # {"dataset_name": "amgut1_log2_standard_scaled",
    #     "dataframe": get_df_from_path("/home/da2343/cs685_fall22/data/amgut1_data_log2_standard_scaled_transformed.csv")},

    # {"dataset_name": "amgut1_box_cox",
    #     "dataframe": get_df_from_path("/home/da2343/cs685_fall22/data/amgut1_data_box_cox_transformed.csv")},


    # {"dataset_name": "amgut2",
    #     "dataframe": get_df_from_path("/home/da2343/cs685_fall22/data/amgut2_data_power_transformed.csv")},

    # {"dataset_name": "crohns",
    #     "dataframe": get_df_from_path("/home/da2343/cs685_fall22/data/crohns_data_power_transformed.csv")},

    # {"dataset_name": "ioral",
    #  "dataframe": get_df_from_path("/home/da2343/cs685_fall22/data/ioral_data_power_transformed.csv")},

    # {"dataset_name": "hmp2prot",
    #  "dataframe": get_df_from_path("/home/da2343/cs685_fall22/data/hmp2prot_data_power_transformed.csv")},

    # {"dataset_name": "hmp216S",
    #  "dataframe": get_df_from_path("/home/da2343/cs685_fall22/data/hmp216S_data_power_transformed.csv")},

    # {"dataset_name": "baxter_crc",
    #     "dataframe": get_df_from_path("/home/da2343/cs685_fall22/data/baxter_crc_data_power_transformed.csv")},

    # {"dataset_name": "glne007",
    #  "dataframe": get_df_from_path("/home/da2343/cs685_fall22/data/glne007_data_power_transformed.csv")},
]
params_df_list = []

for dataset in dataset_list:
    dataset_name = dataset["dataset_name"]
    df = dataset["dataframe"]
    n_row, n_col = df.shape

    # arrange n_row into 10 folds
    n_fold = 10  # TODO: change to 5
    n_row_fold = int(n_row/n_fold)
    n_row_fold_vec = np.ones(n_fold, dtype=int)*n_row_fold
    n_row_fold_vec[:n_row % n_fold] += 1
    n_row_fold_cumsum = np.cumsum(n_row_fold_vec)

    n_col_list = np.arange(n_col)

    params_dict = {
        'Dataset': [dataset_name],
        '# of Samples': n_row_fold_cumsum,
        'Index of Prediction Col': n_col_list,
    }

    params_df = pd.MultiIndex.from_product(
        params_dict.values(),
        names=params_dict.keys()
    ).to_frame().reset_index(drop=True)

    params_df_list.append(params_df)

params_concat_df = pd.concat(params_df_list, ignore_index=True)


n_tasks, ncol = params_concat_df.shape
date_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
job_name = f"model_comparison_{date_time}"
job_dir = "/scratch/da2343/" + job_name
results_dir = os.path.join(job_dir, "results")
os.system("mkdir -p "+results_dir)
params_concat_df.to_csv(os.path.join(job_dir, "params.csv"), index=False)

run_one_contents = f"""#!/bin/bash
#SBATCH --array=0-{n_tasks-1}
#SBATCH --time=4:00:00
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --error={job_dir}/slurm-%A_%a.out
#SBATCH --output={job_dir}/slurm-%A_%a.out
#SBATCH --job-name={job_name}
cd {job_dir}
python run_one.py $SLURM_ARRAY_TASK_ID
"""
run_one_sh = os.path.join(job_dir, "run_one.sh")
with open(run_one_sh, "w") as run_one_f:
    run_one_f.write(run_one_contents)

run_orig_py = "demo_run.py"
run_one_py = os.path.join(job_dir, "run_one.py")
shutil.copyfile(run_orig_py, run_one_py)
orig_dir = os.path.dirname(run_orig_py)
orig_results = os.path.join(orig_dir, "results")
os.system("mkdir -p "+orig_results)
orig_csv = os.path.join(orig_dir, "params.csv")
params_concat_df.to_csv(orig_csv, index=False)

msg = f"""created params CSV files and job scripts, test with
python {run_orig_py}
SLURM_ARRAY_TASK_ID=0 bash {run_one_sh}"""
print(msg)
