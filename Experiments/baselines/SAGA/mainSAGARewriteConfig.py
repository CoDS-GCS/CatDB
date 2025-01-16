from argparse import ArgumentParser
import yaml
import re
import pandas as pd


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--dataset-path', type=str, default=None)
    args = parser.parse_args()

    if args.metadata_path is None:
        raise Exception("--metadata-path is a required parameter!")

    # read .yaml file and extract values:
    with open(args.metadata_path, "r") as f:
        try:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
            args.dataset_name = config_data[0].get('name')
            args.target_attribute = config_data[0].get('dataset').get('target')
            args.task_type = config_data[0].get('dataset').get('type')
            try:
                args.data_source_train_path = args.dataset_path + "/" + config_data[0].get('dataset').get('train').replace("{user}/", "")
                args.data_source_test_path = args.dataset_path + "/" + config_data[0].get('dataset').get('test').replace("{user}/", "")
            except Exception as ex:
                raise Exception(ex)

        except yaml.YAMLError as ex:
            raise Exception(ex)

    return args


if __name__ == '__main__':
    args = parse_arguments()
    df_test = pd.read_csv(args.data_source_test_path, na_values=[' ', '?', '-'])
    df_test = df_test.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    cols = df_test.columns
    config_strs = [f"- name: {args.dataset_name}",
                       "  dataset:",
                       f"    train: \'{args.dataset_name}/{args.dataset_name}_aug_SAGA_train.csv\'",
                       f"    test: \'{args.dataset_name}/{args.dataset_name}_orig_test.csv\'",
                       f"    target_table: {args.dataset_name}",
                       f"    target: '{cols[-1]}'",
                       f"    type: {args.task_type}"
                       "\n"]
    config_str = "\n".join(config_strs)

    yaml_file_local = f'{args.dataset_path}/{args.dataset_name}/{args.dataset_name}_aug_SAGA.yaml'
    f_local = open(yaml_file_local, 'w')
    f_local.write("--- \n \n")
    f_local.write(config_str)
    f_local.close()

