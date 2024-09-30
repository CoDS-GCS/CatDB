from argparse import ArgumentParser
from catalog.Catalog import load_data_source_profile
import pandas as pd
import yaml


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default=None)
    parser.add_argument('--data-profile-path', type=str, default=None)
    parser.add_argument('--log-file-name', type=str, default=None)
    parser.add_argument('--statistic-file-name', type=str, default=None)

    args = parser.parse_args()

    if args.log_file_name is None:
        raise Exception("--log-file-name is a required parameter!")

    if args.statistic_file_name is None:
        raise Exception("--statistic-file-name is a required parameter!")

    if args.data_profile_path is None:
        raise Exception("--data-profile-path is a required parameter!")

    return args


if __name__ == '__main__':
    args = parse_arguments()
    catalog = load_data_source_profile(data_source_path=args.data_profile_path,
                                                file_format="JSON",
                                                target_attribute=None,
                                                enable_reduction=False,
                                                categorical_values_restricted_size=-1)
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

    try:
        df_1 = pd.read_csv(args.log_file_name)
    except Exception as err:
        df_1 = pd.DataFrame(columns=["dataset_name", "number_bool", "number_int", "number_float", "number_string"])

    df_1.loc[len(df_1)] = [args.dataset_name, nbools, nints, nfloats, nstrings]
    df_1.to_csv(args.log_file_name, index=False)

    try:
        df_2 = pd.read_csv(args.statistic_file_name)

    except Exception as err:
        df_2 = pd.DataFrame(columns=["dataset_name", "col_index", "data_type", "missing_values_count","total_values_count", "distinct_values_count","number_rows"])

    index = 0
    for k in profile_info.keys():
        pi = profile_info[k]
        df_2.loc[index] = [args.dataset_name, index + 1, pi.short_data_type, pi.missing_values_count,
                               pi.total_values_count - pi.missing_values_count, pi.distinct_values_count, catalog.nrows]
        index += 1

    df_2 = df_2.sort_values(by=['missing_values_count'], ascending=True)
    df_2["col_index"] = [i for i in range(1, index+1)]
    df_2.to_csv(args.statistic_file_name, index=False)
