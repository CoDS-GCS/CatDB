import sys
import openml
from sklearn.model_selection import train_test_split
from pathlib import Path
from argparse import ArgumentParser
from DatasetPrepare import split_data_save
from DatasetPrepare import get_metadata
from DatasetPrepare import rename_col_names
from DatasetPrepare import save_config


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
    
    # datasetIDs = [ (45693,'simulated_electricity','binary', 1),
    #                (23513,'KDD98','binary', 2),                  
    #                (45570,'Higgs','binary', 3),
    #                (45072,'airlines','binary', 4),
    #                (40514,'BNG_credit_g','binary', 5),
    #                (45579,'Microsoft','multiclass', 6),
    #                (45056,'cmc','multiclass', 7),
    #                (37,'diabetes','multiclass', 8),
    #                (43476,'3-million-Sudoku-puzzles-with-ratings','multiclass', 9),
    #                (155,'pokerhand','multiclass', 10),
    #                (4549,'Buzzinsocialmedia_Twitter','regression', 11),
    #                (45045,'delays_zurich_transport','regression', 12),
    #                (44065,'nyc-taxi-green-dec-2016','regression', 13),
    #                (44057,'black_friday','regression', 14),
    #                (42080,'federal_election','regression', 15),
    #               ]
    

    datasetIDs = [ (37,'diabetes','multiclass', 8)]    
   
    dataset_list = 'row,orig_dataset_name,dataset_name,nrows,ncols,file_format,task_type,number_classes,original_url,target_feature,description\n'
    script_list =""
    for (dataset_id,dataset_name,task_type, dataset_index) in datasetIDs:        
        print(f" Downloading Dataset: dataset name={dataset_name}, dataset ID={dataset_id} \n")

        dataset = openml.datasets.get_dataset(dataset_id, download_qualities=False)
        data, y, categorical_indicator, attribute_names = dataset.get_data()
        target_attribute = dataset.default_target_attribute

         # Split and save original dataset
        nrows, ncols, number_classes = get_metadata(data=data, target_attribute=target_attribute)
        split_data_save(data=data, ds_name=dataset_name,out_path= args.data_out_path)
        save_config(dataset_name=dataset_name, target=target_attribute, task_type=task_type, data_out_path=args.data_out_path, setting_out_path=args.setting_out_path)

        # Split and rename dataset-name, then save  
        dataset_out_name = f"oml_dataset_{dataset_index}"
        split_data_save(data=data, ds_name=dataset_out_name, out_path= args.data_out_path)
        save_config(dataset_name=dataset_out_name, target=target_attribute, task_type=task_type, data_out_path=args.data_out_path, setting_out_path=args.setting_out_path)

        # Rename cols and dataset name, then split and save it
        dataset_out_name = f"oml_dataset_{dataset_index}_rnc"
        target_attribute, nrows, ncols, number_classes = rename_col_names(data=data, ds_name=dataset_out_name, target_attribute=target_attribute, out_path=args.data_out_path)
        save_config(dataset_name=dataset_out_name, target=target_attribute, task_type=task_type, data_out_path=args.data_out_path, setting_out_path=args.setting_out_path)  