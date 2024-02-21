from argparse import ArgumentParser
from catalog.Catalog import load_data_source_profile
from prompt.PromptBuilder import prompt_factory
from llm.GenerateLLMCode import GenerateLLMCode
from util.Config import PROMPT_FUNC
import pandas as pd
import yaml


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--data-profile-path', type=str, default=None)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--llm-model', type=str, default=None)
    args = parser.parse_args()

    if args.metadata_path is None:
        raise Exception("--metadata-path is a required parameter!")

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
                args.data_source_train_path = config_data[0].get('dataset').get('train').replace("{user}/", "")
                args.data_source_test_path = config_data[0].get('dataset').get('test').replace("{user}/", "")
            except Exception as ex:
                raise Exception(ex)

            try:
                args.number_folds = int(config_data[0].get('folds'))
            except yaml.YAMLError as ex:
                args.number_folds = 1

        except yaml.YAMLError as ex:
            raise Exception(ex)

    return args


if __name__ == '__main__':
    args = parse_arguments()
    catalog = load_data_source_profile(data_source_path=args.data_profile_path,
                                       file_format="JSON",
                                       target_attribute=args.target_attribute)
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

    df_1 = pd.DataFrame(
        columns=["dataset", "prompt_representation_type", "prompt_example_type", "prompt_number_example"
            , "number_tokens", "number_bool", "number_int", "number_float", "number_string"])
    log_path = f"{args.output_path}/statistics_1.csv"
    llm = GenerateLLMCode(model=args.llm_model)
    for rt in PROMPT_FUNC.keys():
        prompt = prompt_factory(catalog=catalog,
                                representation_type=rt,
                                example_type="Random",
                                number_example=0,
                                task_type=args.task_type,
                                number_iteration=10,
                                target_attribute=args.target_attribute,
                                data_source_train_path=args.data_source_train_path,
                                data_source_test_path=args.data_source_test_path,
                                number_folds=args.number_folds)

        # Generate LLM code
        prompt_format = prompt.format(examples=None)
        prompt_rule = prompt_format["rules"]
        prompt_msg = prompt_format["question"]
        ntokens = llm.get_number_tokens(prompt_rules=prompt_rule, prompt_message=prompt_msg)
        df_1.loc[len(df_1)] = [args.dataset_name, rt, "Random", 0, ntokens, nbools, nints, nfloats, nstrings]


    df_1.to_csv(f"{args.output_path}/statistics_1.csv", index=False)

    log_path = f"{args.output_path}/statistics_2.csv"
    df_2 = pd.DataFrame(columns=["dataset", "col_index", "data_type", "missing_values_count","total_values_count",
                     "distinct_values_count","number_rows"])

    index = 0
    for k in profile_info.keys():
        pi = profile_info[k]
        df_2.loc[index] = [args.dataset_name, index + 1, pi.short_data_type, pi.missing_values_count,
                               pi.total_values_count - pi.missing_values_count, pi.distinct_values_count, catalog.nrows]
        index += 1

    df_2 = df_2.sort_values(by=['missing_values_count'], ascending=True)
    df_2.to_csv(log_path, index=False)
