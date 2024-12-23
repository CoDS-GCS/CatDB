from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import os
from datasets import get_root_data_path, get_root_catalog_path
import re
import numpy as np
import yaml


def _split_data_save(data: pd.DataFrame, ds_name, out_path, target_table: str = None, write_data: bool = True):
    if target_table is None:
        target_table = ds_name
    data_train, data_test = train_test_split(data, test_size=0.3, random_state=42)
    _, data_verify = train_test_split(data_train, test_size=0.1, random_state=42)
    Path(f"{out_path}/{ds_name}").mkdir(parents=True, exist_ok=True)

    if write_data:
        data.to_csv(f'{out_path}/{ds_name}/{target_table}.csv', index=False)
    data_train.to_csv(f'{out_path}/{ds_name}/{target_table}_train.csv', index=False)
    data_test.to_csv(f'{out_path}/{ds_name}/{target_table}_test.csv', index=False)
    data_verify.to_csv(f'{out_path}/{ds_name}/{target_table}_verify.csv', index=False)


def _get_metadata(data, target_attribute):
    (nrows, ncols) = data.shape
    n_classes = data[target_attribute].nunique()
    number_classes = f'{n_classes}'

    return nrows, ncols, number_classes


def _refactor_openml_description(description):
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


def _save_config(dataset_name, target, task_type, data_out_path, description=None, multi_table: bool = False,
                 target_table: str = None):
    if target_table is None:
        target_table = dataset_name

    config_strs = [f"- name: {dataset_name}",
                   "  dataset:",
                   f"    multi_table: {multi_table}",
                   f"    train: \'{dataset_name}/{target_table}_train.csv\'",
                   f"    test: \'{dataset_name}/{target_table}_test.csv\'",
                   f"    verify: \'{dataset_name}/{target_table}_verify.csv\'",
                   f"    target_table: {target_table}",
                   f"    target: '{target}'",
                   f"    type: {task_type}"
                   "\n"]
    config_str = "\n".join(config_strs)

    yaml_file_local = f'{data_out_path}/{dataset_name}/{dataset_name}.yaml'
    f_local = open(yaml_file_local, 'w')
    f_local.write("--- \n \n")
    f_local.write(config_str)
    f_local.close()

    if description is None:
        des = ""
    else:
        des = _refactor_openml_description(description=description)

    description_file = f'{data_out_path}/{dataset_name}/{dataset_name}.txt'
    f = open(description_file, 'w')
    f.write(des)
    f.close()
    return yaml_file_local


def _load_dependency_info(dependency_file: str, datasource_name: str):
    with open(dependency_file, "r") as f:
        try:
            dep = yaml.load(f, Loader=yaml.FullLoader)
            ds_name = dep[0].get('name')
            if ds_name == datasource_name:
                tbls = dict()
                for k, v in dep[0].get('tables').items():
                    FKs = dep[0].get('tables').get(k).get('FK')
                    d = []
                    if FKs is not None:
                        d = FKs.split(",")

                    tbls[k] = d

                return tbls

        except yaml.YAMLError as ex:
            raise Exception(ex)
        except:
            return None


def prepare_dataset(path: str, task_type: str, target_attribute: str, out_path: str=None, description: str= None, name: str = None):
    if path.endswith('/'):
        path = path[:-1]

    if name is not None and path is not None:
        absolute_fname = f"{path}/{name}.csv"
    elif path is not None and name is None:
        if os.path.isdir(path):
            raise Exception("Dataset Name is required! set dataset name: 'name=...'")
        else:
            absolute_fname = path
    else:
        raise Exception("Dataset Name or Path is required! set dataset name and path: 'name=...' 'path=...'")

    if os.path.exists(absolute_fname):
        base_name = os.path.basename(path)
        name, _ = os.path.splitext(base_name)
    else:
        raise Exception(f"Dataset is not exist: {absolute_fname}")

    data = pd.read_csv(absolute_fname, low_memory=False, encoding='UTF-8')
    data = data[data[target_attribute].notna()]
    if out_path is None:
        out_path = get_root_data_path()
    _split_data_save(data=data, ds_name=name, out_path=out_path, target_table=name, write_data=True)
    metadata_path = _save_config(dataset_name=name, target=target_attribute, task_type=task_type, data_out_path=out_path,
                 description=description, target_table=name,  multi_table=False)

    return {"dataset_name": name,
            "source_dataset_path": absolute_fname,
            "root_data_path": out_path,
            "root_catalog_path": f"{get_root_catalog_path()}/{name}",
            "data_profile_path": f"{get_root_catalog_path()}/{name}/data_profile",
            "metadata_path": metadata_path,
            "data": data}