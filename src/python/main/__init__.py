import os
import sys
from os.path import dirname

CATDB_PACKAGE_PATH = dirname(__file__)
if CATDB_PACKAGE_PATH not in sys.path:
    sys.path.append(CATDB_PACKAGE_PATH)

from util import config, load_args
from util.DatasetPrepare import prepare_dataset
from datasets import get_dataset_names, get_dataset_path, get_catalogs, get_catalog_path
from catalog.Catalog import load_data_source_profile
from catalog.Dependency import load_dependency_info
from pipegen.GeneratePipeLine import generate_and_verify_pipeline
from util.Config import load_config
from ui import create_report
import time

args = None
time_catalog: float = 0
load_catalog: bool = False


def _load_settings(name: str, config):
    global args

    args = load_args(name=name, PACKAGE_PATH=CATDB_PACKAGE_PATH, cfg=config)
    load_config(system_log=args.system_log, llm_model=args.llm_model, rules_path=args.rules_path, evaluation_acc=False,
                config_path=args.config_path, api_config_path=args.APIKeys_File,
                data_cleaning_rules_path=args.data_cleaning_rules_path)

    # check the data clean is available:
    if os.path.isfile(args.data_source_train_clean_path):
        args.data_source_train_path = args.data_source_train_clean_path

    if os.path.isfile(args.data_source_test_clean_path):
        args.data_source_test_path = args.data_source_test_clean_path

    if os.path.isfile(args.data_source_verify_clean_path):
        args.data_source_verify_path = args.data_source_verify_clean_path


def load_dataset_catalog(name: str, config) -> list:
    global args
    global time_catalog
    global load_catalog

    load_catalog = True

    args = load_args(name=name, PACKAGE_PATH=CATDB_PACKAGE_PATH, cfg=config)
    catalog = []
    load_config(system_log=args.system_log, llm_model=args.llm_model, rules_path=args.rules_path, evaluation_acc=False,
                config_path=args.config_path, api_config_path=args.APIKeys_File,
                data_cleaning_rules_path=args.data_cleaning_rules_path)

    # check the data clean is available:
    if os.path.isfile(args.data_source_train_clean_path):
        args.data_source_train_path = args.data_source_train_clean_path

    if os.path.isfile(args.data_source_test_clean_path):
        args.data_source_test_path = args.data_source_test_clean_path

    if os.path.isfile(args.data_source_verify_clean_path):
        args.data_source_verify_path = args.data_source_verify_clean_path

    time_start = time.time()
    catalog.append(load_data_source_profile(data_source_path=args.data_profile_path,
                                            file_format="JSON",
                                            target_attribute=args.target_attribute,
                                            enable_reduction=args.enable_reduction,
                                            categorical_values_restricted_size=-1))

    time_end = time.time()
    time_catalog = time_end - time_start
    return catalog


def generate_pipeline(catalog: list, config):
    from util.Config import __execute_mode, __gen_verify_mode, __sub_task_data_preprocessing, \
        __sub_task_feature_engineering, __sub_task_model_selection

    if load_catalog:
        name = args.dataset_name
        dependency_file = f"{args.catalog_path}/dependency.yaml"
        dependencies = load_dependency_info(dependency_file=dependency_file, datasource_name=name)
    else:
        name = catalog["dataset_name"]
        config["root_data_path"] = catalog["root_data_path"]
        config["root_catalog_path"] = catalog["root_catalog_path"]
        config["data_profile_path"] = catalog["data_profile_path"]
        config["metadata_path"] = catalog["metadata_path"]
        catalog = catalog["data"]
        dependencies = None
    _load_settings(name=name, config=config)

    ti = 0
    t = args.prompt_number_iteration
    begin_iteration = 1
    end_iteration = 1

    while begin_iteration < args.prompt_number_iteration + end_iteration:
        if args.prompt_representation_type == "CatDBChain":
            final_status, code = generate_and_verify_pipeline(args=args, catalog=catalog, run_mode=__gen_verify_mode,
                                                              sub_task=__sub_task_data_preprocessing,
                                                              time_catalog=time_catalog,
                                                              iteration=begin_iteration, dependency=dependencies)
            if final_status:
                final_status, code = generate_and_verify_pipeline(args=args, catalog=catalog,
                                                                  run_mode=__gen_verify_mode,
                                                                  sub_task=__sub_task_feature_engineering,
                                                                  previous_result=code,
                                                                  time_catalog=time_catalog, iteration=begin_iteration,
                                                                  dependency=dependencies)
                if final_status:
                    final_status, code = generate_and_verify_pipeline(args=args, catalog=catalog,
                                                                      run_mode=__execute_mode,
                                                                      sub_task=__sub_task_model_selection,
                                                                      previous_result=code,
                                                                      time_catalog=time_catalog,
                                                                      iteration=begin_iteration,
                                                                      dependency=dependencies)
                    if final_status:
                        begin_iteration += 1

        else:
            final_status, code = generate_and_verify_pipeline(args=args, catalog=catalog, run_mode=__execute_mode,
                                                              time_catalog=time_catalog, iteration=begin_iteration,
                                                              dependency=dependencies)
            if final_status:
                begin_iteration += 1

        ti += 1
        if ti > t:
            break
    args.result_format = "pipeline"
    return vars(args)


__all__ = [
    "config",
    "load_dataset_catalog",
    "get_dataset_names",
    "get_dataset_path",
    "get_catalogs",
    "get_catalog_path",
    "generate_pipeline",
    "create_report",
    "prepare_dataset"
]
