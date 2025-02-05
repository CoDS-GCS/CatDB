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
    args = parser.parse_args()
    
    if args.data_out_path is None:
        raise Exception("--data-out-path is a required parameter!")       
    
    return args


if __name__ == '__main__':
    args = parse_arguments()    
    
    datasetIDs = [  #(15,"Breast-w","binary",2),
                    # (23,"CMC","multiclass",3),
                    # (31,"Credit-g","binary",4),
                    # (37,"Diabetes","binary",5),
                    # (50,"Tic-Tac-Toe","binary",6),         
                    # (42132,"Traffic","multiclass",19),
                    # (1509,"Walking-Activity","multiclass",20),
                    # (44048,"Bike-Sharing","regression",22),
                    # (44051,"House-Sales","regression",23),
                    # (44065,"NYC","regression",24),
                    # (1486,"Nomao","binary",33),
                    # (1476,"Gas-Drift","multiclass",34),
                    # (41166,"Volkert","multiclass",35),
                    # (42438,"Titanic","binary",36),
                    # (45068,"Adult","binary",37),
                    # (42031,"Iris","multiclass",38),
                    #(1457,"Amazon","multiclass",200),
                    #(4139,"Wikidata","binary",201),
                    #(42775,"KDDCup09","binary",202),
                    #(1594,"News","binary",203),
                    # (41165,"Robert","multiclass",204),
                    # (41161,"Riccardo","binary",205),
                    # (41163,"Dilbert","multiclass",206),
                    #(42570,"Mercedes","regression",207),
                    #(422,"Topo","regression",208),
                    #(42571,"Allstate","regression",209), 
                    (42343,"KDD98","binary",210),  
                    (1111,"KDDCup09","binary",211)
                    
                ]
   
    dataset_list =  pd.DataFrame(columns=["Row","ID","dataset_name", "orig_name","nrows","ncols","nclasses","target"])
    
    script_list_1 =""
    script_list_2 =""

    for (dataset_id,dataset_name,task_type, dataset_index) in datasetIDs:        
        print(f" Downloading Dataset: dataset name={dataset_name}, dataset ID={dataset_id} \n")

        dataset = openml.datasets.get_dataset(dataset_id, download_qualities=False)
        dataset_description = dataset.description
        data, y, categorical_indicator, attribute_names = dataset.get_data()
        target_attribute = dataset.default_target_attribute

        n_classes = data[dataset.default_target_attribute].nunique()           

         # Split and save original dataset
        nrows, ncols, number_classes = get_metadata(data=data, target_attribute=target_attribute)
        split_data_save(data=data, ds_name=dataset_name,out_path= args.data_out_path)
        save_config(dataset_name=dataset_name, target=target_attribute, task_type=task_type, data_out_path=args.data_out_path, description=dataset_description)
        dataset_out_name = dataset_name

        # Split and rename dataset-name, then save  
        # dataset_out_name = f"oml_dataset_{dataset_index}"
        # split_data_save(data=data, ds_name=dataset_out_name, out_path= args.data_out_path)
        # save_config(dataset_name=dataset_out_name, target=target_attribute, task_type=task_type, data_out_path=args.data_out_path, setting_out_path=args.setting_out_path)

        # Rename cols and dataset name, then split and save it
        # dataset_out_name = f"oml_dataset_{dataset_index}_rnc"
        # target_attribute, nrows, ncols, number_classes = rename_col_names(data=data, ds_name=dataset_out_name, target_attribute=target_attribute, out_path=args.data_out_path)
        # save_config(dataset_name=dataset_out_name, target=target_attribute, task_type=task_type, data_out_path=args.data_out_path, description=dataset_description) 

        dataset_list.loc[len(dataset_list)] = [dataset_index, dataset_id, dataset_out_name, dataset_name, nrows, ncols, number_classes, target_attribute]
        dataset_list.to_csv(f"{args.data_out_path}/dataset_list.csv") 

        script_list_1 += f"$CMD {dataset_name} {task_type} # {dataset_name}\n"
        script_list_2 += f"$CMD {dataset_out_name} {task_type} # {dataset_name}\n"


    # print(script_list_1)
    # print(script_list_2)    