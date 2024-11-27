import os
import sys
from os.path import dirname

CATDB_PACKAGE_PATH = dirname(__file__)
if CATDB_PACKAGE_PATH not in sys.path:
    sys.path.append(CATDB_PACKAGE_PATH)

from catalog.Catalog import load_data_source_profile
from catalog.Dependency import load_dependency_info
from pipegen.GeneratePipeLine import generate_and_verify_pipeline, run_pipeline, clean_categorical_data
from util.FileHandler import read_text_file_line_by_line
from util.Config import load_config
from pipegen.Metadata import Metadata
from util import load_args
import time
import yaml


args = None


def load_dataset_catalog(name: str) -> list:
    global args

    args = load_args(name=name)
    catalog = []
    time_start = time.time()
    dependency_file = f"{args.catalog_path}/dependency.yaml"
    dependencies = load_dependency_info(dependency_file=dependency_file, datasource_name=args.dataset_name)

    load_config(system_log=args.system_log, llm_model=args.llm_model, rules_path=args.rules_path, evaluation_acc=False)

    # check the data clean is available:
    if os.path.isfile(args.data_source_train_clean_path):
        args.data_source_train_path = args.data_source_train_clean_path

    if os.path.isfile(args.data_source_test_clean_path):
        args.data_source_test_path = args.data_source_test_clean_path

    if os.path.isfile(args.data_source_verify_clean_path):
        args.data_source_verify_path = args.data_source_verify_clean_path

    catalog.append(load_data_source_profile(data_source_path=args.data_profile_path,
                                            file_format="JSON",
                                            target_attribute=args.target_attribute,
                                            enable_reduction=args.enable_reduction,
                                            categorical_values_restricted_size=-1))

    time_end = time.time()
    time_catalog = time_end - time_start

    print(time_catalog)
    return catalog