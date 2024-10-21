import sys
import openml
from sklearn.model_selection import train_test_split
from pathlib import Path
from argparse import ArgumentParser
from DatasetPrepare import split_data_save
from DatasetPrepare import get_metadata
from DatasetPrepare import rename_col_names
from DatasetPrepare import save_config
import pandas as pd
import numpy as np
import random


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--data-in-path', type=str, default=None) 
    parser.add_argument('--data-out-path', type=str, default=None)   
    args = parser.parse_args()
    
    if args.data_out_path is None:
        raise Exception("--data-out-path is a required parameter!")       
    
    return args

# Function to introduce outliers
def introduce_outliers(df, outlier_percentage=0.01, outlier_factor=10, label_col_index = 0):
    df_with_outliers = df.copy()
    n_samples, n_features = df_with_outliers.shape
    n_outlier_samples = int(np.floor(outlier_percentage * n_samples * n_features))
    
    outlier_indices = np.random.choice(n_samples * n_features, n_outlier_samples, replace=False)
    for index in outlier_indices:
        row = index // n_features
        col = index % n_features        
        if col != label_col_index and np.issubdtype(df_with_outliers.dtypes[col], np.number):
            df_with_outliers.iat[row, col] *= outlier_factor
    
    return df_with_outliers



# Function to introduce missing values
def introduce_missing_values(df, col, missing_percentage=0.1):
    df_with_missing = df.copy()
    n_samples,_ = df_with_missing.shape
    n_missing_samples = int(np.floor(missing_percentage * n_samples))
    
    missing_indices = random.sample(range(0, n_samples), n_missing_samples)
    
    for row in missing_indices:            
        df_with_missing.iat[row, col] = np.nan
    
    return df_with_missing

if __name__ == '__main__':
    args = parse_arguments() 
    datasetIDs = [
    #              ("NYC", "tip_amount", "regression",53),
                  #("Volkert", "class", "multiclass",54),
                  ("Utility", "CSRI", "regression",55)
                  ]

    missing_percentages = [0.1, 0.2, 0.3, 0.4, 0.5]
    outliers_percentage = [0.01, 0.02, 0.03, 0.04, 0.05]
    selected_outlier = 0.05

    script_list_1 =""
    script_list_2 =""
    script_list_3 =""    
     
    for (dataset_name, target_attribute,task_type, dataset_index) in datasetIDs:     
        df = pd.read_csv(f"{args.data_in_path}/{dataset_name}.csv") 
        label_col_index = -1 

        for cols_percentage in [1]:
            n_features = len(df.columns)
            index = 0
            feature_indices = []
            for col in df.columns:
                if col == target_attribute:
                    label_col_index = index
                    index += 1                    
                    continue
                feature_indices.append(index)
                index += 1       

        for outlier_percentage in outliers_percentage:
            df_tmp = introduce_outliers(df=df, outlier_percentage=outlier_percentage, outlier_factor=10, label_col_index=label_col_index)

            df_tmp = df_tmp[df_tmp[target_attribute].notna()]
            dataset_out_name = f"{dataset_name}-out-{outlier_percentage}-np-0-nc-0-mv-0"
            split_data_save(data=df_tmp, ds_name=dataset_out_name,out_path= args.data_out_path, target_table=dataset_out_name, write_data=True)
            save_config(dataset_name=dataset_out_name, target=target_attribute, task_type=task_type, data_out_path=args.data_out_path, description="")
            script_list_1 += f"$CMD {dataset_out_name} {task_type} # {dataset_name}\n"        


        missing_col_indices = random.sample(range(0, len(feature_indices)), int(cols_percentage * len(feature_indices)))     
        for perc in missing_percentages:
                df_tmp = df.copy()
                for colin in missing_col_indices:
                    df_tmp = introduce_missing_values(df=df_tmp, col=colin, missing_percentage=perc)

                # Rename cols and dataset name, then split and save it
                dataset_out_name = f"{dataset_name}-out-0-np-{cols_percentage}-nc-{len(missing_col_indices)}-mv-{perc}"
                split_data_save(data=df_tmp, ds_name=dataset_out_name,out_path= args.data_out_path, target_table=dataset_out_name, write_data=True)
                save_config(dataset_name=dataset_out_name, target=target_attribute, task_type=task_type, data_out_path=args.data_out_path, description="")
                
                script_list_2 += f"$CMD {dataset_out_name} {task_type} # {dataset_name}\n"

                ###############
                for outlier_percentage in [selected_outlier]:
                    df_tmp_both = introduce_outliers(df=df_tmp, outlier_percentage=outlier_percentage, outlier_factor=10, label_col_index=label_col_index)
                    dataset_out_name = f"{dataset_name}-out-{outlier_percentage}-np-{cols_percentage}-nc-{len(missing_col_indices)}-mv-{perc}"
                    split_data_save(data=df_tmp, ds_name=dataset_out_name,out_path= args.data_out_path, target_table=dataset_out_name, write_data=True)
                    save_config(dataset_name=dataset_out_name, target=target_attribute, task_type=task_type, data_out_path=args.data_out_path, description="")
                    script_list_3 += f"$CMD {dataset_out_name} {task_type} # {dataset_name}\n"



    print(script_list_1)
    print(script_list_2)
    print(script_list_3)
