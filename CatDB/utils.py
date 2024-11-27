"""This module implements load dataset related functions"""

import os
from os.path import dirname
import pandas as pd
from typing import List
from argparse import ArgumentParser
import yaml


def _get_dataset_metadata_path(name: str) -> str:
    """
    Given a dataset name, output the file path.
    """
    
    module_path = dirname(__file__)
    path = f"{module_path}/data/{name}/{name}.yaml"
    return path

def _get_root_data_path() -> str:
    """
    Given a dataset name, output the file path.
    """
    
    module_path = dirname(__file__)
    path = f"{module_path}/data/"
    return path

def _get_dataset_catalog_path(name: str) ->str:
    """
    Given a dataset name, output the file path.
    """
    
    module_path = dirname(__file__)
    path = f"{module_path}/catalog/{name}"
    return path

def _get_output_path(name: str) ->str:
    """
    Given a dataset name, output the file path.
    """
    
    module_path = dirname(__file__)
    path = f"{module_path}/catdb-results/{name}"
    return path

def _get_API_key_path() ->str:
    """
    Given a dataset name, output the file path.
    """
    
    module_path = dirname(__file__)
    path = f"{module_path}/APIKeys.yaml"
    return path

def _save_text_file(fname: str, data):
    try:
        f = open(fname, 'w')
        f.write(data)
        f.close()
    except Exception as ex:
        raise Exception (f"Error in save file:\n {ex}")
    
def set_config(model: str, API_key: str, iteration: int = 1, error_iteration: int = 15, reduction: bool = True, setting: str = "CatDB",
               output_path: str = "results.csv", error_path: str = "error.csv", system_log: str = "system.log"):
   
    module_path = dirname(__file__)

    API_key_template = """
---

- llm_platform: {}
  key_1: '{}'    
"""

    setting_txt = f"""
---

- setting: {setting}
  prompt_number_iteration: {iteration}
  prompt_number_iteration_error: {error_iteration}
  llm_model: {model}
  reduction: {reduction}
  result_output_path: {module_path}/{output_path}
  error_output_path: {module_path}/{error_path}
  system_log: {module_path}/{system_log}
"""

    platform = None
    if model in {"gpt-4o", " gpt-4-turbo"}:
        platform = "OpenAI"
    elif model in {"llama3-70b-8192", "llama-3.1-70b-versatile"}:
        platform = "Meta"
    elif model in {"gemini-1.5-pro-latest"}:
        platform = "Google"

    ak = API_key_template.format(platform, API_key)

    path_key = f"{module_path}/.APIKeys.yaml"
    path_setting = f"{module_path}/.setting.yaml"
    _save_text_file(data=ak, fname=path_key)
    _save_text_file(data=setting_txt, fname=path_setting)
    

def load_args(name: str):
    module_path = dirname(__file__)
    parser = ArgumentParser()
    parser.add_argument('--default', type=str, required=False, default=None)
    parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
    args = parser.parse_args()  
    args.metadata_path = _get_dataset_metadata_path(name=name)
    args.root_data_path = _get_root_data_path()
    args.catalog_path = _get_dataset_catalog_path(name=name)
    args.APIKeys_File=_get_API_key_path()
    args.output_path = _get_output_path(name=name)
    args.data_profile_path = f"{args.catalog_path}/data_profile"
    args.prompt_samples_type = 'Random'
    args.prompt_number_samples = 0
    args.description = ''
    args.dataset_description = 'No'   

    with open(f"{module_path}/setting.yaml", "r") as f:
        try:
            setting_data = yaml.load(f, Loader=yaml.FullLoader)
            args.prompt_representation_type = setting_data[0].get('setting')
            args.prompt_number_iteration = setting_data[0].get('prompt_number_iteration')
            args.prompt_number_iteration_error = setting_data[0].get('prompt_number_iteration_error')
            args.llm_model = setting_data[0].get('llm_model')
            args.result_output_path = setting_data[0].get('result_output_path')
            args.error_output_path = setting_data[0].get('error_output_path')
            args.system_log = setting_data[0].get('system_log')
            args.enable_reduction = setting_data[0].get('reduction')
            
        except yaml.YAMLError as ex:
            raise Exception(ex)

    # read .yaml file and extract values:
    with open(args.metadata_path, "r") as f:
        try:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
            args.dataset_name = config_data[0].get('name')
            args.target_attribute = config_data[0].get('dataset').get('target')
            args.task_type = config_data[0].get('dataset').get('type')
            args.multi_table = config_data[0].get('dataset').get('multi_table')
            args.target_table = config_data[0].get('dataset').get('target_table')
            if args.multi_table is None or args.multi_table not in {True, False}:
                args.multi_table = False

            try:
                args.data_source_path = f"{args.root_data_path}/{args.dataset_name}/{args.dataset_name}.csv"
                args.data_source_train_path = f"{args.root_data_path}/{config_data[0].get('dataset').get('train')}"
                args.data_source_test_path = f"{args.root_data_path}/{config_data[0].get('dataset').get('test')}"
                args.data_source_verify_path = f"{args.root_data_path}/{config_data[0].get('dataset').get('verify')}"
                args.data_source_train_clean_path = f"{args.data_source_train_path.replace('.csv','')}_clean.csv"
                args.data_source_test_clean_path = f"{args.data_source_test_path.replace('.csv','')}_clean.csv"
                args.data_source_verify_clean_path = f"{args.data_source_verify_path.replace('.csv','')}_clean.csv"
                args.data_source_clean_path = f"{args.data_source_path.replace('.csv', '')}_clean.csv"

            except Exception as ex:
                raise Exception(ex)

        except yaml.YAMLError as ex:
            raise Exception(ex)   

    return args