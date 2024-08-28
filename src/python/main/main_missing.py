from argparse import ArgumentParser
from catalog.Catalog import load_data_source_profile
from prompt.PromptBuilder import prompt_factory_missing_values
from llmdataprepare.DataPrepareLLM import DataPrepareLLM
from runcode.RunCode import RunCode
from util.FileHandler import save_prompt
from util.FileHandler import save_text_file, read_text_file_line_by_line
from util.Config import load_config, load_missing_value_dataset, convert_df_to_string
from util.LogResults import save_log
from util.ErrorResults import ErrorResults
from pipegen.Metadata import Metadata
import  pandas as pd
import time
import datetime
import yaml
import os


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--root-data-path', type=str, default=None)
    parser.add_argument('--data-profile-path', type=str, default=None)
    parser.add_argument('--dataset-description', type=str, default="yes")
    parser.add_argument('--prompt-representation-type', type=str, default=None)
    parser.add_argument('--prompt-samples-type', type=str, default=None)
    parser.add_argument('--prompt-number-samples', type=int, default=0)
    parser.add_argument('--prompt-number-request-samples', type=int, default=1)
    parser.add_argument('--prompt-number-iteration', type=int, default=1)
    parser.add_argument('--prompt-number-iteration-error', type=int, default=1)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--llm-model', type=str, default=None)
    parser.add_argument('--enable-reduction', type=bool, default=False)
    parser.add_argument('--result-output-path', type=str, default="/tmp/results.csv")
    parser.add_argument('--error-output-path', type=str, default="/tmp/catdb_error.csv")
    parser.add_argument('--run-code', type=bool, default=False)
    parser.add_argument('--system-log', type=str, default="/tmp/catdb-system-log.dat")

    args = parser.parse_args()

    if args.metadata_path is None:
        raise Exception("--metadata-path is a required parameter!")

    if args.root_data_path is None:
        raise Exception("--root-data-path is a required parameter!")

    if args.data_profile_path is None:
        raise Exception("--data-profile-path is a required parameter!")

    # read .yaml file and extract values:
    with open(args.metadata_path, "r") as f:
        try:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
            args.dataset_name = config_data[0].get('name')
            args.target_attribute = config_data[0].get('dataset').get('target')
            args.task_type = config_data[0].get('dataset').get('type')
            try:
                args.data_source_path = f"{args.root_data_path}/{args.dataset_name}/{args.dataset_name}.csv"
                args.data_source_train_path = f"{args.root_data_path}/{config_data[0].get('dataset').get('train')}"
                args.data_source_test_path = f"{args.root_data_path}/{config_data[0].get('dataset').get('test')}"
                args.data_source_verify_path = f"{args.root_data_path}/{config_data[0].get('dataset').get('verify')}"
            except Exception as ex:
                raise Exception(ex)

        except yaml.YAMLError as ex:
            raise Exception(ex)

    if args.prompt_samples_type is None:
        raise Exception("--prompt-sample-type is a required parameter!")

    if args.prompt_number_samples is None:
        raise Exception("--prompt-number-samples is a required parameter!")

    if args.llm_model is None:
        raise Exception("--llm-model is a required parameter!")

    if args.prompt_number_iteration is None:
        args.prompt_number_iteration = 1

    if args.prompt_representation_type in {"CatDB", "CatDBChain"}:
        args.enable_reduction = True

    if args.dataset_description.lower() == "yes":
        dataset_description_path = args.metadata_path.replace(".yaml", ".txt")
        args.description = read_text_file_line_by_line(fname=dataset_description_path)
        args.dataset_description = 'Yes'
    else:
        args.description = None
        args.dataset_description = 'No'

    return args


def missing_value_imputation(args, data, catalog):
    from util.Config import _missing_value_train_data_samples, _missing_value_train_data

    nsamples_request = args.prompt_number_request_samples
    for i in range(0, len(data), nsamples_request):
        target_samples_size = min(i+nsamples_request, len(data))
        tmp_df = data[i: target_samples_size]
        results = dict()
        cols = list(tmp_df.columns[tmp_df.isna().any()].values)
        target_samples = convert_df_to_string(df=tmp_df, row_prefix="### Row")
        prompt = prompt_factory_missing_values(catalog=catalog,
                                representation_type=args.prompt_representation_type,
                                number_samples=_missing_value_train_data_samples,
                                samples_missed_values=_missing_value_train_data,
                                columns_has_missing_values=cols,
                                dataset_description=args.description,
                                target_attribute=args.target_attribute,
                                target_samples=target_samples,
                                target_samples_size = target_samples_size)

        prompt_format = prompt.format()
        prompt_system_message = prompt_format["system_message"]
        prompt_user_message = prompt_format["user_message"]
        schema_data = prompt_format["schema_data"]

        # Save prompt:
        prompt_file_name = f"{args.llm_model}-{prompt.class_name}-{args.dataset_description}"
        file_name = f'{args.output_path}/{prompt_file_name}'
        prompt_fname = f"{file_name}.prompt"
        save_prompt(fname=prompt_fname, system_message=prompt_system_message, user_message=prompt_user_message)

        result, prompt_token_count, time_tmp_gen = DataPrepareLLM.data_prepare_llm(user_message=prompt_user_message,
                                                                                   system_message=prompt_system_message)
        print(result)
        break

if __name__ == '__main__':
    from util.Config import __execute_mode, __gen_verify_mode, __sub_task_data_preprocessing, \
        __sub_task_feature_engineering, __sub_task_model_selection

    args = parse_arguments()
    load_config(system_log=args.system_log, llm_model=args.llm_model, rules_path="RulesMissingValue.yaml")

    begin_iteration = args.prompt_number_iteration
    end_iteration = 1
    time_start = time.time()
    catalog = load_data_source_profile(data_source_path=args.data_profile_path,
                                       file_format="JSON",
                                       target_attribute=args.target_attribute,
                                       enable_reduction=args.enable_reduction)

    time_end = time.time()
    time_catalog = time_end - time_start
    ti = 0
    t = args.prompt_number_iteration * 2

    prompt_representation_type_orig = args.prompt_representation_type
    missed_value_cols = []

    if len(catalog.columns_numerical_missing_values) > 0:
        missed_value_cols.extend(catalog.columns_numerical_missing_values)

    if len(catalog.columns_bool_missing_values) > 0:
        missed_value_cols.extend(catalog.columns_bool_missing_values)

    if len(catalog.columns_categorical_missing_values) > 0:
        missed_value_cols.extend(catalog.columns_categorical_missing_values)

    if len(catalog.columns_others_missing_values) > 0:
        missed_value_cols.extend(catalog.columns_others_missing_values)

    if len(missed_value_cols) > 0:
        df = pd.read_csv(args.data_source_path)
        load_missing_value_dataset(data=df, target_attribute=args.target_attribute, task_type=args.task_type,
                                   number_samples=args.prompt_number_samples)
        na_data = df[df.isna().any(axis=1)]

        missing_value_imputation(args=args, catalog=catalog, data=na_data)

    # print(missed_value_cols)
