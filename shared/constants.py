import pandas as pd



def get_df_from_path(path):
    df = pd.read_csv(path, header=0)
    return df


root_data_dir = "/projects/genomic-ml/da2343/ml_project_1/data"

'''
dataset_list = [
    
    {"dataset_name": "Dec22_all_power",
        "dataframe": get_df_from_path(f"{root_data_dir}/necromass/Dec22_bacteria_fungi_conservative_r_all_power_transformed.csv")},
 
    {"dataset_name": "Dec22_same_soil_power",
        "dataframe": get_df_from_path(f"{root_data_dir}/necromass/Dec22_bacteria_fungi_conservative_r_same_soil_power_transformed.csv")},
    
    {"dataset_name": "Dec22_same_melan_high_power",
        "dataframe": get_df_from_path(f"{root_data_dir}/necromass/Dec22_bacteria_fungi_conservative_r_same_melan_high_power_transformed.csv")},
    
    {"dataset_name": "Dec22_same_melan_low_power",
        "dataframe": get_df_from_path(f"{root_data_dir}/necromass/Dec22_bacteria_fungi_conservative_r_same_melan_low_power_transformed.csv")},
    
    {"dataset_name": "Dec22_all_log",
        "dataframe": get_df_from_path(f"{root_data_dir}/necromass/Dec22_bacteria_fungi_conservative_r_all_log_transformed.csv")},
    
    {"dataset_name": "Dec22_same_soil_log",
        "dataframe": get_df_from_path(f"{root_data_dir}/necromass/Dec22_bacteria_fungi_conservative_r_same_soil_log_transformed.csv")},
    
    {"dataset_name": "Dec22_same_melan_high_log",
        "dataframe": get_df_from_path(f"{root_data_dir}/necromass/Dec22_bacteria_fungi_conservative_r_same_melan_high_log_transformed.csv")},
    
    {"dataset_name": "Dec22_same_melan_low_log",
        "dataframe": get_df_from_path(f"{root_data_dir}/necromass/Dec22_bacteria_fungi_conservative_r_same_melan_low_log_transformed.csv")},
    ]

dataset_dict = {    
    "Dec22_all_power": f"{root_data_dir}/necromass/Dec22_bacteria_fungi_conservative_r_all_power_transformed.csv",
    "Dec22_same_soil_power": f"{root_data_dir}/necromass/Dec22_bacteria_fungi_conservative_r_same_soil_power_transformed.csv",
    "Dec22_same_melan_high_power": f"{root_data_dir}/necromass/Dec22_bacteria_fungi_conservative_r_same_melan_high_power_transformed.csv",
    "Dec22_same_melan_low_power": f"{root_data_dir}/necromass/Dec22_bacteria_fungi_conservative_r_same_melan_low_power_transformed.csv",
    "Dec22_all_log": f"{root_data_dir}/necromass/Dec22_bacteria_fungi_conservative_r_all_log_transformed.csv",
    "Dec22_same_soil_log": f"{root_data_dir}/necromass/Dec22_bacteria_fungi_conservative_r_same_soil_log_transformed.csv",
    "Dec22_same_melan_high_log": f"{root_data_dir}/necromass/Dec22_bacteria_fungi_conservative_r_same_melan_high_log_transformed.csv",
    "Dec22_same_melan_low_log": f"{root_data_dir}/necromass/Dec22_bacteria_fungi_conservative_r_same_melan_low_log_transformed.csv"
}
'''

dataset_list = [
    
    ## Necromass datasets with species level data
    # {"dataset_name": "necromass_bacteria_species",
    #     "dataframe": get_df_from_path(f"{root_data_dir}/necromass/bacteria_rarefied_otu_mapping_PKfixTrimmed_power_transformed.csv")},
 
    # {"dataset_name": "necromass_fungi_species",
    #     "dataframe": get_df_from_path(f"{root_data_dir}/necromass/fungi_rarefied_otu_mapping_PKfix_power_transformed.csv")},
    
    # {"dataset_name": "necromass_bacteria_fungi_species",
    #     "dataframe": get_df_from_path(f"{root_data_dir}/necromass/bacteria_fungi_rarefied_otu_mapping_PKfixTrimmed_power_transformed.csv")},
    
    ## Necromass datasets with genus level data
    # {"dataset_name": "necromass_bacteria_genus",
    #     "dataframe": get_df_from_path(f"{root_data_dir}/necromass/bacteria_genus_power_transformed.csv")},
    
    # {"dataset_name": "necromass_fungi_genus",
    #     "dataframe": get_df_from_path(f"{root_data_dir}/necromass/fungi_genus_power_transformed.csv")},
    
    # {"dataset_name": "necromass_bacteria_fungi_genus",
    #     "dataframe": get_df_from_path(f"{root_data_dir}/necromass/bacteria_fungi_genus_power_transformed.csv")},
    
    ## Necromass datasets with conservative level data
    # {"dataset_name": "necromass_bacteria_conservative",
    #     "dataframe": get_df_from_path(f"{root_data_dir}/necromass/bacteria_conservative_power_transformed.csv")},
    
    # {"dataset_name": "necromass_fungi_conservative",
    #     "dataframe": get_df_from_path(f"{root_data_dir}/necromass/fungi_conservative_power_transformed.csv")},
    
    # {"dataset_name": "necromass_bacteria_fungi_conservative",
    #     "dataframe": get_df_from_path(f"{root_data_dir}/necromass/bacteria_fungi_conservative_power_transformed.csv")},
    

    {"dataset_name": "amgut1",
        "dataframe": get_df_from_path(f"{root_data_dir}/amgut1_data_power_transformed.csv")},
    
    # {"dataset_name": "amgut1_standard_scaled",
    #     "dataframe": get_df_from_path(f"{root_data_dir}/amgut1_data_standard_scaled.csv")},

    # {"dataset_name": "amgut1_log1_standard_scaled",
    #     "dataframe": get_df_from_path(f"{root_data_dir}/amgut1_data_log1_standard_scaled_transformed.csv")},

    {"dataset_name": "crohns",
        "dataframe": get_df_from_path(f"{root_data_dir}/crohns_data_power_transformed.csv")},

    {"dataset_name": "ioral",
     "dataframe": get_df_from_path(f"{root_data_dir}/ioral_data_power_transformed.csv")},
    
    {"dataset_name": "amgut2",
        "dataframe": get_df_from_path(f"{root_data_dir}/amgut2_data_power_transformed.csv")},

    {"dataset_name": "hmp2prot",
     "dataframe": get_df_from_path(f"{root_data_dir}/hmp2prot_data_power_transformed.csv")},

    {"dataset_name": "hmp216S",
     "dataframe": get_df_from_path(f"{root_data_dir}/hmp216S_data_power_transformed.csv")},

    {"dataset_name": "baxter_crc",
        "dataframe": get_df_from_path(f"{root_data_dir}/baxter_crc_data_power_transformed.csv")},

    {"dataset_name": "glne007",
     "dataframe": get_df_from_path(f"{root_data_dir}/glne007_data_power_transformed.csv")},
]



dataset_dict = {
    "necromass_bacteria_species": f"{root_data_dir}/necromass/bacteria_rarefied_otu_mapping_PKfixTrimmed_power_transformed.csv",
    "necromass_fungi_species": f"{root_data_dir}/necromass/fungi_rarefied_otu_mapping_PKfix_power_transformed.csv",
    "necromass_bacteria_fungi_species": f"{root_data_dir}/necromass/bacteria_fungi_rarefied_otu_mapping_PKfixTrimmed_power_transformed.csv",
    
    "necromass_bacteria_genus": f"{root_data_dir}/necromass/bacteria_genus_power_transformed.csv",
    "necromass_fungi_genus": f"{root_data_dir}/necromass/fungi_genus_power_transformed.csv",
    "necromass_bacteria_fungi_genus": f"{root_data_dir}/necromass/bacteria_fungi_genus_power_transformed.csv",
    
    "necromass_bacteria_conservative": f"{root_data_dir}/necromass/bacteria_conservative_power_transformed.csv",
    "necromass_fungi_conservative": f"{root_data_dir}/necromass/fungi_conservative_power_transformed.csv",
    "necromass_bacteria_fungi_conservative": f"{root_data_dir}/necromass/bacteria_fungi_conservative_power_transformed.csv",
    
    "amgut1": f"{root_data_dir}/amgut1_data_power_transformed.csv",
    "amgut1_standard_scaled": f"{root_data_dir}/amgut1_data_standard_scaled.csv",
    "amgut1_log1_standard_scaled": f"{root_data_dir}/amgut1_data_log1_standard_scaled_transformed.csv",
    "amgut2": f"{root_data_dir}/amgut2_data_power_transformed.csv",
    "crohns": f"{root_data_dir}/crohns_data_power_transformed.csv",
    "baxter_crc": f"{root_data_dir}/baxter_crc_data_power_transformed.csv",
    "enterotype": f"{root_data_dir}/enterotype_data_power_transformed.csv",
    "esophagus": f"{root_data_dir}/esophagus_data_power_transformed.csv",
    "glne007": f"{root_data_dir}/glne007_data_power_transformed.csv",
    "global_patterns": f"{root_data_dir}/global_patterns_data_power_transformed.csv",
    "hmp2prot": f"{root_data_dir}/hmp2prot_data_power_transformed.csv",
    "hmp216S": f"{root_data_dir}/hmp216S_data_power_transformed.csv",
    'ioral': f'{root_data_dir}/ioral_data_power_transformed.csv',
    'mixmpln': f'{root_data_dir}/mixmpln_real_data_power_transformed.csv',
    'soilrep': f'{root_data_dir}/soilrep_data_power_transformed.csv',
}