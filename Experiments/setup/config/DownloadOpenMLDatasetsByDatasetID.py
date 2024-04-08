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


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--data-out-path', type=str, default=None)
    parser.add_argument('--setting-out-path', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.data_out_path is None:
        raise Exception("--data-out-path is a required parameter!")    
    
    if args.setting_out_path is None:
        raise Exception("--setting-out-path is a required parameter!")
    
    return args


if __name__ == '__main__':
    args = parse_arguments()    
    
    datasetIDs = [(45570,"Higgs","binary",1),
              (41147,"albert","binary",2),
              (1218,"Click_prediction_small","binary",3),
              (43489,"Census-Augmented","binary",4),
              (267,"BNG_heart-statlog","binary",5),
              (1110,"KDDCup99_full","multiclass",6),
              (42803,"road-safety","multiclass",7),
              (43044,"drug-directory","multiclass",8),
              (42734,"okcupid-stem","multiclass",9),
              (1509,"walking-activity","multiclass",10),
              (45274,"PASS","regression",11),
              (42396,"aloi","regression",12),
              (45049,"MD_MIX_Mini_Copy","regression",13),
              (41167,"dionis","regression",14),
              (44320,"Meta_Album_BRD_Extended","regression",15)]
   
    #dataset_list = 'row,orig_dataset_name,dataset_name,nrows,ncols,file_format,task_type,number_classes,original_url,target_feature,description\n'
    dataset_list =  pd.DataFrame(columns=["Row","ID","dataset_name", "orig_name","nrows","ncols","nclasses","target"])
    
    script_list =""
    for (dataset_id,dataset_name,task_type, dataset_index) in datasetIDs:        
        print(f" Downloading Dataset: dataset name={dataset_name}, dataset ID={dataset_id} \n")

        dataset = openml.datasets.get_dataset(dataset_id, download_qualities=False)
        dataset_description = dataset.description
        data, y, categorical_indicator, attribute_names = dataset.get_data()
        target_attribute = dataset.default_target_attribute

        n_classes = data[dataset.default_target_attribute].nunique()
        # if n_classes == 2:
        #         task_type = "binary"
        # elif n_classes < 300 :
        #      task_type = "multiclass"
        # else:
        #         task_type = "regression"            

         # Split and save original dataset
        nrows, ncols, number_classes = get_metadata(data=data, target_attribute=target_attribute)
        split_data_save(data=data, ds_name=dataset_name,out_path= args.data_out_path)
        save_config(dataset_name=dataset_name, target=target_attribute, task_type=task_type, data_out_path=args.data_out_path, setting_out_path=args.setting_out_path, description=dataset_description)
        dataset_out_name = dataset_name

        # # Split and rename dataset-name, then save  
        # dataset_out_name = f"oml_dataset_{dataset_index}"
        # split_data_save(data=data, ds_name=dataset_out_name, out_path= args.data_out_path)
        # save_config(dataset_name=dataset_out_name, target=target_attribute, task_type=task_type, data_out_path=args.data_out_path, setting_out_path=args.setting_out_path)

        # Rename cols and dataset name, then split and save it
        # dataset_out_name = f"oml_dataset_{dataset_index}_rnc"
        # target_attribute, nrows, ncols, number_classes = rename_col_names(data=data, ds_name=dataset_out_name, target_attribute=target_attribute, out_path=args.data_out_path)
        # save_config(dataset_name=dataset_out_name, target=target_attribute, task_type=task_type, data_out_path=args.data_out_path, setting_out_path=args.setting_out_path) 

        # "Row","ID","dataset_name", "orig_name","nrows","ncols","nclasses","target"
        dataset_list.loc[len(dataset_list)] = [dataset_index, dataset_id, dataset_out_name, dataset_name, nrows, ncols, number_classes, target_attribute]
        dataset_list.to_csv(f"{args.data_out_path}/dataset_list.csv") 