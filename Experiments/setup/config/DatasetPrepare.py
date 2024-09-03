import sys
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
from argparse import ArgumentParser
import re
import numpy as np

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--dataset-root-path', type=str, default=None)
    parser.add_argument('--dataset-name', type=str, default=None)
    parser.add_argument('--target-attribute', type=str, default=None)
    parser.add_argument('--task-type', type=str, default=None)
    parser.add_argument('--target-table', type=str, default=None)
    parser.add_argument('--multi-table', type=str, default=None)
    parser.add_argument('--dataset-description', type=str, default="")
    parser.add_argument('--data-out-path', type=str, default=None)    
    
    args = parser.parse_args()
    return args


def split_data_save(data, ds_name, out_path):
    data_train, data_test = train_test_split(data, test_size=0.3, random_state=42)
    _, data_verify =  train_test_split(data_train, test_size=0.1, random_state=42)
    Path(f"{out_path}/{ds_name}").mkdir(parents=True, exist_ok=True)

    data.to_csv(f'{out_path}/{ds_name}/{ds_name}.csv', index=False)
    data_train.to_csv(f'{out_path}/{ds_name}/{ds_name}_train.csv', index=False)
    data_test.to_csv(f'{out_path}/{ds_name}/{ds_name}_test.csv', index=False)
    data_verify.to_csv(f'{out_path}/{ds_name}/{ds_name}_verify.csv', index=False)

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


def refactor_openml_description(description):
    """Refactor the description of an openml dataset to remove the irrelevant parts."""
    if description is None:
        return None
    splits = re.split("\n", description)
    blacklist = [
        "Please cite",
        "Author",
        "Source",
        "Author:",
        "Source:",
        "Please cite:",
    ]
    sel = ~np.array(
        [
            np.array([blacklist_ in splits[i] for blacklist_ in blacklist]).any()
            for i in range(len(splits))
        ]
    )
    description = str.join("\n", np.array(splits)[sel].tolist())

    splits = re.split("###", description)
    blacklist = ["Relevant Papers"]
    sel = ~np.array(
        [
            np.array([blacklist_ in splits[i] for blacklist_ in blacklist]).any()
            for i in range(len(splits))
        ]
    )
    description = str.join("\n\n", np.array(splits)[sel].tolist())
    return description


def save_config(dataset_name,target, task_type, data_out_path, description=None, multi_table: bool=False, target_table: str=None):
    if target_table is None:
        target_table=dataset_name
    config_strs = [f"- name: {dataset_name}",
                       "  dataset:",
                       f"    multi_table: {multi_table}",
                       f"    train: \'{dataset_name}/{dataset_name}_train.csv\'",
                       f"    test: \'{dataset_name}/{dataset_name}_test.csv\'",
                       f"    verify: \'{dataset_name}/{dataset_name}_verify.csv\'",
                       f"    target_table: {target_table}",
                       f"    target: {target}",
                       f"    type: {task_type}"
                       "\n"]
    config_str = "\n".join(config_strs)

    yaml_file_local = f'{data_out_path}/{dataset_name}/{dataset_name}.yaml'
    f_local = open(yaml_file_local, 'w')
    f_local.write("--- \n \n")
    f_local.write(config_str)
    f_local.close() 

    des = description
    if description is None:
        des = ""
    else:
        des = refactor_openml_description(description=description)    
        
    description_file = f'{data_out_path}/{dataset_name}/{dataset_name}.txt'
    f = open(description_file, 'w')
    f.write(des)
    f.close()


if __name__ == '__main__':
    args = parse_arguments()
    
    # Read dataset
    data = pd.read_csv(f"{args.dataset_root_path}/{args.dataset_name}/{args.target_table}.csv")

    # Split and save original dataset
    nrows, ncols, number_classes = get_metadata(data=data, target_attribute=args.target_attribute)
    split_data_save(data=data, ds_name=args.dataset_name,out_path= args.data_out_path)
    save_config(dataset_name=args.dataset_name, target=args.target_attribute, task_type=args.task_type, data_out_path=args.data_out_path, description=args.dataset_description, target_table=args.target_table, multi_table=args.multi_table)
