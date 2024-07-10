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


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default=None) 
    parser.add_argument('--data-out-path', type=str, default=None)   
    args = parser.parse_args()
    
    if args.data_out_path is None:
        raise Exception("--data-out-path is a required parameter!")       
    
    return args


# Function to introduce missing values
def introduce_missing_values(df, col, missing_percentage=0.1):
    df_with_missing = df.copy()
    n_samples, n_features = df_with_missing.shape
    n_missing_samples = int(np.floor(missing_percentage * n_samples))
    
    missing_indices = np.random.choice(n_samples, n_missing_samples, replace=False)
    for index in missing_indices:
        row = index // n_features
        df_with_missing.iat[row, col] = np.nan
    
    return df_with_missing

if __name__ == '__main__':
    # args = parse_arguments() 
    # df = pd.read_csv(args.dataset_path)
    data_out_path = "GeneratedDatasets"
    datasetIDs = [("adult", "income", "binary",50),
                  ("bank", "y", "binary",51),
                  ("br2000", "a14", "binary",52)]

    missing_percentages = [0.1, 0.20, 0.30]
    script_list_2 =""
     
    for (dataset_name, target_attribute,task_type, dataset_index) in datasetIDs:     
        df = pd.read_csv(f"Datasets/{dataset_name}.csv")
        
        for cols_percentage in [0.2, 0.4, 0.6, 0.8, 1]:
            n_features = len(df.columns)
            index = 0
            feature_indices = []
            for col in df.columns:
                if col == target_attribute:
                    index += 1
                    continue
                feature_indices.append(index)
                index += 1


            missing_col_indices = np.random.choice(feature_indices, size=int(cols_percentage * len(feature_indices)))       
            
            for perc in missing_percentages:
                index = 0
                df_tmp = df.copy()
                for colin in missing_col_indices:
                    df_tmp = introduce_missing_values(df=df_tmp, col=index, missing_percentage=perc)

                # df_tmp.to_csv(f"GeneratedDatasets/{dataset_name}-nc-{len(missing_col_indices)}-mv-{perc}.csv")
                ########################################
                # Rename cols and dataset name, then split and save it
                dataset_out_name = f"gen_dataset_{dataset_index}-np-{cols_percentage}-nc-{len(missing_col_indices)}-mv-{perc}_rnc"
                target_attribute_rn, nrows, ncols, number_classes = rename_col_names(data=df_tmp, ds_name=dataset_out_name, target_attribute=target_attribute, out_path=data_out_path)
                save_config(dataset_name=dataset_out_name, target=target_attribute_rn, task_type=task_type, data_out_path=data_out_path, description="") 

                script_list_2 += f"$CMD {dataset_out_name} {task_type} # {dataset_name}\n"


    print(script_list_2)
