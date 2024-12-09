from pathlib import Path
from os.path import dirname
from argparse import ArgumentParser
from datasets import get_dataset_metadata_path, get_root_data_path, get_catalog_path
import yaml
import os
import shutil

CATDB_PACKAGE_PATH = None

def _get_rules_path() -> str:
    path = f"{CATDB_PACKAGE_PATH}/Rules.yaml"
    return path


def _get_data_cleaning_rules_path() -> str:
    path = f"{CATDB_PACKAGE_PATH}/RulesDataCleaning.yaml"
    return path


def _get_config_path() -> str:
    path = f"{CATDB_PACKAGE_PATH}/Config.yaml"
    return path


def _get_output_path(name: str, output_path: str) -> str:
    path = f"{output_path}/catdb-results/{name}"
    Path(f"{output_path}/catdb-results").mkdir(parents=True, exist_ok=True)
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _save_text_file(fname: str, data):
    try:
        f = open(fname, 'w')
        f.write(data)
        f.close()
    except Exception as ex:
        raise Exception(f"Error in save file:\n {ex}")


def config(model: str, API_key: str, iteration: int = 1, error_iteration: int = 15, reduction: bool = True,
               setting: str = "CatDB", output_path: str = None, clean_log: bool= True ):

    cfg = dict()

    if output_path is None:
        output_path = '/tmp/catdb-log'
    if clean_log:
        if os.path.exists(output_path) and os.path.isdir(output_path):
            shutil.rmtree(output_path)

    cfg["output_path"] = output_path

    Path(output_path).mkdir(parents=True, exist_ok=True)

    API_key_template = """
---

- llm_platform: {}
  key_1: '{}'    
"""

    cfg["setting"] = setting
    cfg["prompt_number_iteration"] = iteration
    cfg["prompt_number_iteration_error"] = error_iteration
    cfg["llm_model"] = model
    cfg["reduction"] = reduction
    cfg["result_output_path"] = f"{output_path}/results.csv"
    cfg["error_output_path"] = f"{output_path}/error.csv"
    cfg["system_log"] = f"{output_path}/system.log"

    platform = None
    if model in {"gpt-4o", " gpt-4-turbo"}:
        platform = "OpenAI"
    elif model in {"llama3-70b-8192", "llama-3.1-70b-versatile"}:
        platform = "Meta"
    elif model in {"gemini-1.5-pro-latest"}:
        platform = "Google"

    ak = API_key_template.format(platform, API_key)

    path_key = f"{output_path}/APIKeys.yaml"
    _save_text_file(data=ak, fname=path_key)
    cfg["api_config_path"] = path_key
    return cfg


def load_args(name: str, cfg, PACKAGE_PATH: str):
    global CATDB_PACKAGE_PATH
    CATDB_PACKAGE_PATH = PACKAGE_PATH

    parser = ArgumentParser()
    parser.add_argument('--default', type=str, required=False, default=None)
    parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
    args = parser.parse_args()
    args.metadata_path = get_dataset_metadata_path(name=name)
    args.root_data_path = get_root_data_path()
    args.catalog_path = get_catalog_path(name=name)
    args.data_profile_path = f"{args.catalog_path}/data_profile"
    args.prompt_samples_type = 'Random'
    args.prompt_number_samples = 0
    args.description = ''
    args.dataset_description = 'No'
    args.prompt_representation_type = cfg['setting']
    args.prompt_number_iteration = cfg['prompt_number_iteration']
    args.prompt_number_iteration_error = cfg['prompt_number_iteration_error']
    args.llm_model = cfg['llm_model']
    args.result_output_path = cfg['result_output_path']
    args.error_output_path = cfg['error_output_path']
    args.system_log = cfg['system_log']
    args.enable_reduction = cfg['reduction']
    args.APIKeys_File = cfg['api_config_path']

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
                args.data_source_train_clean_path = f"{args.data_source_train_path.replace('.csv', '')}_clean.csv"
                args.data_source_test_clean_path = f"{args.data_source_test_path.replace('.csv', '')}_clean.csv"
                args.data_source_verify_clean_path = f"{args.data_source_verify_path.replace('.csv', '')}_clean.csv"
                args.data_source_clean_path = f"{args.data_source_path.replace('.csv', '')}_clean.csv"

            except Exception as ex:
                raise Exception(ex)

        except yaml.YAMLError as ex:
            raise Exception(ex)
    args.rules_path = _get_rules_path()
    args.output_path = _get_output_path(name=name, output_path=cfg["output_path"])
    args.data_cleaning_rules_path = _get_data_cleaning_rules_path()
    args.config_path = _get_config_path()
    return args