from argparse import ArgumentParser
from catalog.Dependency import load_dependency_info
from pipegen.GeneratePipeLine import  clean_categorical_data, clean_data_catalog
from util.Config import load_config
import time
import yaml


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--root-data-path', type=str, default=None)
    parser.add_argument('--catalog-path', type=str, default=None)
    parser.add_argument('--prompt-number-iteration-error', type=int, default=1)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--llm-model', type=str, default=None)
    parser.add_argument('--enable-reduction', type=bool, default=True)
    parser.add_argument('--result-output-path', type=str, default="/tmp/results.csv")
    parser.add_argument('--error-output-path', type=str, default="/tmp/catdb_error.csv")
    parser.add_argument('--system-log', type=str, default="/tmp/catdb-system-log.dat")

    args = parser.parse_args()

    if args.metadata_path is None:
        raise Exception("--metadata-path is a required parameter!")

    if args.root_data_path is None:
        raise Exception("--root-data-path is a required parameter!")

    if args.catalog_path is None:
        raise Exception("--catalog-path is a required parameter!")

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

    if args.llm_model is None:
        raise Exception("--llm-model is a required parameter!")

    return args


if __name__ == '__main__':
    args = parse_arguments()

    begin_iteration = 1
    end_iteration = 1
    data_profile_path = f"{args.catalog_path}/data_profile"
    time_start = time.time()
    dependency_file = f"{args.catalog_path}/dependency.yaml"
    dependencies = load_dependency_info(dependency_file=dependency_file, datasource_name=args.dataset_name)

    load_config(system_log=args.system_log, llm_model=args.llm_model, rules_path="Rules.yaml")

    # clean_data_catalog(args=args, data_profile_path=data_profile_path, time_catalog=0, iteration=begin_iteration)

    # check the data clean is available:
    clean_categorical_data(args=args, data_profile_path=data_profile_path, time_catalog=0, iteration=begin_iteration)
