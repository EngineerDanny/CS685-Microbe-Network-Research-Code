import pandas as pd
import numpy as np

source_target_path = "/projects/genomic-ml/da2343/ml_project_1/model_edges_comp/source_target"

#amgut1
date_time = "2023-06-02_09:14"
# date_time = "2023-06-02_15:10"
## date_time = "2023-06-02_18:38"
# date_time = "2023-06-02_19:12"
date_time = "2023-06-05_13:50"

#crohns
# date_time = "2023-09-13_18:18"
## date_time = "2023-09-13_20:38"

#ioral
date_time = "2023-09-13_18:53"
# date_time = "2023-09-13_21:04"
date_time = "2023-09-21_19:47"


source_target_df = pd.read_csv(f"{source_target_path}/source_target_{date_time}.csv")


algorithm = "GGM"
filtered_algorithm_df = source_target_df[source_target_df["algorithm"] == algorithm]
# Get unique values as list from column name `Dataset`
dataset_list = filtered_algorithm_df["data_set_name"].unique().tolist()
for dataset in dataset_list:
    # Get new dataframe with only the dataset
    sub_dataset_df = filtered_algorithm_df[filtered_algorithm_df["data_set_name"] == dataset]
    n_samples_list = sub_dataset_df["n_total_samples"].unique().tolist()
    fold_id_list = sub_dataset_df["fold_id"].unique().tolist()

    test_error_df_with_samples_list = []
    for n_samples in n_samples_list:
        sub_sub_dataset_df = sub_dataset_df[sub_dataset_df["n_total_samples"] == n_samples]
        filtered_fold_id_df_list = []
        for fold_id  in fold_id_list:
            filtered_fold_id_df = sub_sub_dataset_df[sub_sub_dataset_df["fold_id"] == fold_id]
            # get only the source, target and weight columns
            filtered_fold_id_df = filtered_fold_id_df[["source", "target", "weight", "fold_id"]]
            # find the mean of the weight column using the source and target columns
            filtered_source_target_df = filtered_fold_id_df.groupby(["source", "target", "fold_id"]).mean().reset_index()
            filtered_fold_id_df_list.append({
                'fold_id': fold_id,
                'n_edges': filtered_source_target_df.shape[0]
            })
        local_df = pd.DataFrame(filtered_fold_id_df_list)
        mean_n_edges = local_df["n_edges"].mean()
        std_n_edges = local_df["n_edges"].std()
        median_n_edges = local_df["n_edges"].median()
        min_n_edges = local_df["n_edges"].min()
        max_n_edges = local_df["n_edges"].max()
        
        new_row = {
            "n_total_samples": n_samples,
            "mean_n_edges": mean_n_edges,
            "std_n_edges": std_n_edges,
            "median_n_edges": median_n_edges,
            "min_n_edges": min_n_edges,
            "max_n_edges": max_n_edges,
            "algorithm": algorithm
        }
        test_error_df_with_samples_list.append(pd.DataFrame([new_row]))
    pd.concat(test_error_df_with_samples_list).to_csv(f"{dataset}_{algorithm}_edges_sample.csv", index=False)
    
    


lasso_coef_path = "/projects/genomic-ml/da2343/ml_project_1/model_edges_comp/lasso_coef"
source_target_df = pd.read_csv(f"{lasso_coef_path}/lasso_coef_{date_time}.csv")

algorithm = "LASSO"
filtered_algorithm_df = source_target_df[source_target_df["algorithm"] == algorithm]
dataset_list = filtered_algorithm_df["data_set_name"].unique().tolist()

for dataset in dataset_list:
    # Get new dataframe with only the dataset
    sub_dataset_df = filtered_algorithm_df[filtered_algorithm_df["data_set_name"] == dataset]
    n_samples_list = sub_dataset_df["n_total_samples"].unique().tolist()
    fold_id_list = sub_dataset_df["fold_id"].unique().tolist()
    
    # Create an empty list to store the source, target and weight dataframes
    test_error_df_with_samples_list = []
    for n_samples in n_samples_list:
        sub_sub_dataset_df = sub_dataset_df[sub_dataset_df["n_total_samples"] == n_samples]

        filtered_fold_id_df_list = []
        for fold_id in fold_id_list:
            filtered_fold_id_df = sub_sub_dataset_df[sub_sub_dataset_df["fold_id"] == fold_id]
            reg_param_list = filtered_fold_id_df["reg_param"].unique().tolist()
            best_validation_score = np.inf
            
            for reg_param in reg_param_list:
                filtered_reg_param = filtered_fold_id_df[filtered_fold_id_df["reg_param"] == reg_param]
                subtrain_score = filtered_reg_param['subtrain_score'].mean()
                validation_score = filtered_reg_param['validation_score'].mean()
                index_of_pred_col_list = sorted(filtered_reg_param["index_of_pred_col"].unique().tolist())
                
                coef_matrix_list = []
                for index_of_pred_col in index_of_pred_col_list:
                    filtered_index_of_pred = filtered_reg_param[filtered_reg_param['index_of_pred_col'] == index_of_pred_col]
                    coefs = filtered_index_of_pred['coefs'].values.tolist()
                    coefs_str = coefs[0]
                    # Replace "None" with "nan"
                    coefs_str = coefs_str.replace("None", "nan")
                    coefs_str = coefs_str.replace("\n", "")
                    coefs_str = coefs_str.replace("[", "")
                    coefs_str = coefs_str.replace("]", "")
                    
                    # Convert string to numpy array
                    coefs_arr = np.fromstring(coefs_str, sep=" ")
                    coef_matrix_list.append(coefs_arr)
                    
                # Convert the list of numpy arrays into a matrix
                coefs_mat = np.array(coef_matrix_list)
                # get the indices of the upper and lower triangle elements
                upper_tri = np.triu_indices(coefs_mat.shape[0], k=1)
                lower_tri = np.tril_indices(coefs_mat.shape[0], k=-1)
                # calculate the average of the upper and lower triangle elements
                avg_matrix = (coefs_mat + coefs_mat.T) / 2
                coefs_mat[lower_tri] = avg_matrix[lower_tri]
                coefs_mat[upper_tri] = np.nan
                # replace the diagonal elements with nan
                np.fill_diagonal(coefs_mat, np.nan)
                source_target = np.argwhere(~np.isnan(coefs_mat))
                weights = coefs_mat[source_target[:, 0], source_target[:, 1]]
                source_target_result = [t for t in zip(source_target[:, 0], source_target[:, 1], weights) if np.abs(t[2]) > 0]
                
                # Create a dataframe with source, target and weight columns
                source_target_df = pd.DataFrame(source_target_result, 
                                                columns=["source", "target", "weight"])
             
                if validation_score <= best_validation_score:
                    best_validation_score = validation_score
                    best_reg_param = reg_param
                    best_source_target_df = source_target_df
            
            filtered_fold_id_df_list.append({
                'fold_id': fold_id,
                'n_edges': best_source_target_df.shape[0]
            })
            
        local_df = pd.DataFrame(filtered_fold_id_df_list)
        mean_n_edges = local_df["n_edges"].mean()
        std_n_edges = local_df["n_edges"].std()
        median_n_edges = local_df["n_edges"].median()
        min_n_edges = local_df["n_edges"].min()
        max_n_edges = local_df["n_edges"].max()
            
        new_row = {
            "n_total_samples": n_samples,
            "mean_n_edges": mean_n_edges,
            "std_n_edges": std_n_edges,
            "median_n_edges": median_n_edges,
            "min_n_edges": min_n_edges,
            "max_n_edges": max_n_edges,
            "algorithm": algorithm
        }
        test_error_df_with_samples_list.append(pd.DataFrame([new_row]))
    pd.concat(test_error_df_with_samples_list).to_csv(f"{dataset}_{algorithm}_edges_sample.csv", index=False)
    
    
    
pearson_corr_path = "/projects/genomic-ml/da2343/ml_project_1/model_edges_comp/pearson_corr"
model_complexity_df = pd.read_csv(f"{pearson_corr_path}/pearson_corr_{date_time}.csv")
algorithm_list = model_complexity_df["algorithm"].unique()

for algorithm in ['Pearson', 'Spearman']:
    filtered_algorithm_df = model_complexity_df[model_complexity_df["algorithm"] == algorithm]
    # Get unique values as list from column name `Dataset`
    dataset_list = filtered_algorithm_df["data_set_name"].unique().tolist()
    for dataset in dataset_list:
        # Get new dataframe with only the dataset
        sub_dataset_df = filtered_algorithm_df[filtered_algorithm_df["data_set_name"] == dataset]
        n_samples_list = sub_dataset_df["n_total_samples"].unique().tolist()
        fold_id_list = sub_dataset_df["fold_id"].unique().tolist()
        
        # Create an empty list to store the source, target and weight dataframes
        test_error_df_with_samples_list = []
        for n_samples in n_samples_list:
            sub_sub_dataset_df = sub_dataset_df[sub_dataset_df["n_total_samples"] == n_samples]
            filtered_fold_id_df_list = []
            for fold_id  in fold_id_list:
                test_error_df_list = []
                filtered_fold_id_df = sub_sub_dataset_df[sub_sub_dataset_df["fold_id"] == fold_id]
                reg_param_list = filtered_fold_id_df["reg_param"].unique().tolist()
                for reg_param in reg_param_list:
                    filtered_reg_param = filtered_fold_id_df[filtered_fold_id_df["reg_param"] == reg_param]
                    subtrain_score = filtered_reg_param['subtrain_score'].mean()
                    validation_score = filtered_reg_param['validation_score'].mean()
                    edges = filtered_reg_param['edges'].mean()
            
                    test_error_dict = {
                        'fold_id': fold_id,
                        'subtrain' :  subtrain_score,
                        'validation' : validation_score,
                        'data_set_name': dataset,
                        'reg_param': reg_param,
                        'algorithm' : algorithm,
                        'edges': edges
                    }
                    test_error_df_list.append(pd.DataFrame(test_error_dict, index=[0]))
                test_err_df = pd.concat(test_error_df_list).reset_index()
                # mark only the best reg_param with a blue dot on both subplots
                best_reg_param = test_err_df.loc[test_err_df['validation'].idxmin()]['reg_param']

                source_target_df = pd.read_csv(f"{source_target_path}/source_target_{date_time}.csv")
                filtered_source_target_df = source_target_df[(source_target_df["data_set_name"] == dataset) & 
                                                             (source_target_df["fold_id"] == fold_id) & 
                                                             (source_target_df["algorithm"] == algorithm) & 
                                                             (abs(source_target_df["weight"]) > best_reg_param) &
                                                             (source_target_df["n_total_samples"] == n_samples)
                                                             ]
                filtered_source_target_df = filtered_source_target_df[["source", "target", "weight", "fold_id"]]
                filtered_source_target_df = filtered_source_target_df.groupby(["source", "target", "fold_id"]).mean().reset_index()
                # create a df of fold_id and n_edges
                filtered_fold_id_df_list.append({
                    'fold_id': fold_id,
                    'n_edges': filtered_source_target_df.shape[0]
                    })
            local_df = pd.DataFrame(filtered_fold_id_df_list)
            mean_n_edges = local_df["n_edges"].mean()
            std_n_edges = local_df["n_edges"].std()
            median_n_edges = local_df["n_edges"].median()
            min_n_edges = local_df["n_edges"].min()
            max_n_edges = local_df["n_edges"].max()
            print(algorithm)
            print(local_df)
            
            
            new_row = {
                "n_total_samples": n_samples,
                "mean_n_edges": mean_n_edges,
                "std_n_edges": std_n_edges,
                "median_n_edges": median_n_edges,
                "min_n_edges": min_n_edges,
                "max_n_edges": max_n_edges,
                "algorithm": algorithm
            }
            test_error_df_with_samples_list.append(pd.DataFrame([new_row]))
        pd.concat(test_error_df_with_samples_list).to_csv(f"{dataset}_{algorithm}_edges_sample.csv", index=False)