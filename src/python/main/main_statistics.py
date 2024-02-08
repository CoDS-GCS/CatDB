from argparse import ArgumentParser
from catalog.Catalog import load_data_source_profile
from prompt.PromptBuilder import prompt_factory
from llm.GenerateLLMCode import GenerateLLMCode
import pandas as pd
import os.path
import yaml


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--data-source-path', type=str, default=None)
    parser.add_argument('--data-source-name', type=str, default=None)
    parser.add_argument('--prompt-representation-type', type=str, default=None)
    parser.add_argument('--prompt-example-type', type=str, default=None)
    parser.add_argument('--prompt-number-example', type=int, default=None)
    parser.add_argument('--prompt-number-iteration', type=int, default=None)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--llm-model', type=str, default=None)
    parser.add_argument('--reduction-method', type=str, default=None)
    parser.add_argument('--reduction-size', type=str, default=None)
    parser.add_argument('--suggested-model', type=str, default=None)

    args = parser.parse_args()

    if args.data_source_path is None:
        raise Exception("--data-source-path is a required parameter!")

    if args.data_source_name is None:
        raise Exception("--data-source-name is a required parameter!")

    # read .yaml file and extract values:
    with open(f"{args.data_source_path}/{args.data_source_name}/{args.data_source_name}.yaml", "r") as f:
        try:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
            args.target_attribute = config_data[0].get('dataset').get('target')
            args.task_type = config_data[0].get('dataset').get('type')

            # TODO: read train and test dataset path from yaml file
            args.data_source_train_path = f"data/{args.data_source_name}/{args.data_source_name}_train.csv"
            args.data_source_test_path = f"data/{args.data_source_name}/{args.data_source_name}_test.csv"

            try:
                args.number_folds = int(config_data[0].get('folds'))
            except yaml.YAMLError as excfold:
                args.number_folds = 1

        except yaml.YAMLError as exc:
            raise Exception(exc)

    if args.prompt_example_type is None:
        raise Exception("--prompt-example-type is a required parameter!")

    if args.prompt_number_example is None:
        raise Exception("--prompt-number-example is a required parameter!")

    if args.llm_model is None:
        raise Exception("--llm-model is a required parameter!")

    if args.prompt_number_iteration is None:
        args.prompt_number_iteration = 1

    if args.reduction_method is None:
        args.reduction_method = None

    if args.reduction_size is None:
        args.reduction_size = 0

    if args.prompt_representation_type is None:
        raise Exception("--prompt-representation-type is a required parameter!")

    if args.suggested_model is not None and args.suggested_model != "NA":
        args.suggested_model = f"(such as {args.suggested_model})"
    else:
        args.suggested_model = ""
    return args


if __name__ == '__main__':
    args = parse_arguments()
    profile_info_path = f'{args.data_source_path}/{args.data_source_name}/data_profile_full'
    catalog = load_data_source_profile(data_source_path=profile_info_path,
                                       file_format="JSON",
                                       reduction_method=args.reduction_method,
                                       reduce_size=args.reduction_size,
                                       target_attribute=args.target_attribute)

    prompt = prompt_factory(catalog=catalog,
                            representation_type=args.prompt_representation_type,
                            example_type=args.prompt_example_type,
                            number_example=args.prompt_number_example,
                            task_type=args.task_type,
                            number_iteration=args.prompt_number_iteration,
                            target_attribute=args.target_attribute,
                            data_source_train_path=args.data_source_train_path,
                            data_source_test_path=args.data_source_test_path,
                            suggested_model=args.suggested_model,
                            number_folds=args.number_folds)

    # Generate LLM code
    llm = GenerateLLMCode(model=args.llm_model)
    prompt_format = prompt.format(examples=None)
    prompt_rule = prompt_format["rules"]
    prompt_msg = prompt_format["question"]
    ntokens = llm.get_number_tokens(prompt_rules=prompt_rule, prompt_message=prompt_msg)

    schema_info = catalog.schema_info
    profile_info = catalog.profile_info

    nbools = 0
    nstrings = 0
    nints = 0
    nfloats = 0
    for k in schema_info.keys():
        if schema_info[k] == "bool":
            nbools += 1
        elif schema_info[k] == "str":
            nstrings += 1
        elif schema_info[k] == "float":
            nfloats += 1
        elif schema_info[k] == "int":
            nints += 1

    log_path = f"{args.output_path}/statistics_1.csv"
    flag_exist = os.path.isfile(log_path)
    if flag_exist:
        df_1 = pd.read_csv(log_path, names=["dataset", "prompt_representation_type", "prompt_example_type", "prompt_number_example"
                , "number_tokens", "number_bool", "number_int", "number_float", "number_string"], header=0)
    else:
        df_1 = pd.DataFrame(
            columns=["dataset", "prompt_representation_type", "prompt_example_type", "prompt_number_example"
                , "number_tokens", "number_bool", "number_int", "number_float", "number_string"])

    rep_types = {
        "SCHEMA": "Conf-1",
        "DISTINCT": "Conf-2",
        "MISSING_VALUE": "Conf-3",
        "NUMERIC_STATISTIC": "Conf-4",
        "CATEGORICAL_VALUE": "Conf-5",
        "DISTINCT_MISSING_VALUE": "Conf-6",
        "MISSING_VALUE_NUMERIC_STATISTIC": "Conf-7",
        "MISSING_VALUE_CATEGORICAL_VALUE": "Conf-8",
        "NUMERIC_STATISTIC_CATEGORICAL_VALUE": "Conf-9",
        "ALL": "Conf-10"
    }

    df_1.loc[len(df_1)] = [args.data_source_name, rep_types[args.prompt_representation_type], args.prompt_example_type,
                   args.prompt_number_example, ntokens, nbools, nints, nfloats, nstrings]
    df_1.to_csv(f"{args.output_path}/statistics_1.csv", index=False)

    log_path = f"{args.output_path}/statistics_2.csv"
    flag_exist = os.path.isfile(log_path)
    if not flag_exist:
        df_2 = pd.DataFrame(
            columns=["dataset", "col_index", "data_type", "missing_values_count", "distinct_values_count"])

        index = 0
        for k in profile_info.keys():
            pi = profile_info[k]
            df_2.loc[index] = [args.data_source_name, index + 1, pi.short_data_type, pi.missing_values_count,
                               pi.distinct_values_count]
            index += 1

        df_2.to_csv(log_path, index=False)
