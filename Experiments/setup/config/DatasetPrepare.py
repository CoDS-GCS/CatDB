import sys
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--dataset-info', type=str, default=None)
    parser.add_argument('--data-out-path', type=str, default=None)
    parser.add_argument('--setting-out-path', type=str, default=None)
    
    args = parser.parse_args()

    if args.dataset_info is None:
        raise Exception("--dataset-name is a required parameter!")
    else:
        info = args.dataset_info.split(",")
        args.fname = info[0]
        args.dataset_name = info[1]
        args.dataset_out_name = info[2]
        args.target_attribute = info[3]
        args.task_type = info[4]

    if args.data_out_path is None:
        raise Exception("--data-out-path is a required parameter!")    
    
    if args.setting_out_path is None:
        raise Exception("--setting-out-path is a required parameter!")
    
    return args


def split_data_save(data, ds_name, out_path):
    data_train, data_test = train_test_split(data, test_size=0.3, random_state=42)
    Path(f"{out_path}/{ds_name}").mkdir(parents=True, exist_ok=True)

    data.to_csv(f'{out_path}/{ds_name}/{ds_name}.csv', index=False)
    data_train.to_csv(f'{out_path}/{ds_name}/{ds_name}_train.csv', index=False)
    data_test.to_csv(f'{out_path}/{ds_name}/{ds_name}_test.csv', index=False)

def get_metadata(data, target_attribute):
    (nrows, ncols) = data.shape
    number_classes = 'N/A'
    n_classes = data[target_attribute].nunique()
    number_classes = f'{n_classes}' 

    return nrows, ncols, number_classes

def rename_col_names(data, ds_name, target_attribute, out_path):

    nrows, ncols, number_classes = get_metadata(data=data, target_attribute=target_attribute)

    colnams = data.columns
    new_colnams=dict()
    i = 1
    for col in colnams:
        new_colnams[col]= f"c_{i}"
        i +=1
    data = data.rename(columns=new_colnams)  
    target_attribute = new_colnams[target_attribute] 

    split_data_save(data=data, ds_name=ds_name, out_path=out_path)
    
    return target_attribute, nrows, ncols, number_classes    

def save_config(dataset_name,target, task_type, data_out_path, setting_out_path, description=None):
    config_strs = [f"- name: {dataset_name}",
                       "  dataset:",
                       f"    train: \'{{user}}/data/{dataset_name}/{dataset_name}_train.csv\'",
                       f"    test: \'{{user}}/data/{dataset_name}/{dataset_name}_test.csv\'",
                       f"    target: {target}",
                       f"    type: {task_type}",
                       "  folds: 1",
                       "\n"]
    config_str = "\n".join(config_strs)

    yaml_file_local = f'{data_out_path}/{dataset_name}/{dataset_name}.yaml'
    f_local = open(yaml_file_local, 'w')
    f_local.write("--- \n \n")
    f_local.write(config_str)
    f_local.close()

    yaml_file_benchmark = f'{setting_out_path}/{dataset_name}.yaml'
    f = open(yaml_file_benchmark, 'w')
    f.write("--- \n \n")
    f.write(config_str)
    f.close()   

    if description is not None:
        description_file = f'{data_out_path}/{dataset_name}/{dataset_name}.txt'
        f = open(description_file, 'w')
        f.write(description)
        f.close()

if __name__ == '__main__':
    args = parse_arguments()
    
    # Read dataset
    data = pd.read_csv(f"{args.fname}.csv")

    # Split and save original dataset
    nrows, ncols, number_classes = get_metadata(data=data, target_attribute=args.target_attribute)
    split_data_save(data=data, ds_name=args.dataset_name,out_path= args.data_out_path)
    save_config(dataset_name=args.dataset_name, target=args.target_attribute, task_type=args.task_type, data_out_path=args.data_out_path, setting_out_path=args.setting_out_path)

    # Split and rename dataset-name, then save  
    split_data_save(data=data, ds_name=args.dataset_out_name,out_path= args.data_out_path)
    save_config(dataset_name=args.dataset_out_name, target=args.target_attribute, task_type=args.task_type, data_out_path=args.data_out_path, setting_out_path=args.setting_out_path)

    # Rename cols and dataset name, then split and save it
    dataset_out_name = f"{args.dataset_out_name}_rnc"
    target_attribute, nrows, ncols, number_classes = rename_col_names(data=data, ds_name=dataset_out_name, target_attribute=args.target_attribute, out_path=args.data_out_path)
    save_config(dataset_name=dataset_out_name, target=target_attribute, task_type=args.task_type, data_out_path=args.data_out_path, setting_out_path=args.setting_out_path)   