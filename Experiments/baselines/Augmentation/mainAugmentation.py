from argparse import ArgumentParser
import yaml
from Augmentation import augmentation
import pandas as pd
from pathlib import Path

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--dataset-path', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--source-data', type=str, default=None)

    args = parser.parse_args()

    if args.metadata_path is None:
        raise Exception("--metadata-path is a required parameter!")

    # read .yaml file and extract values:
    with (open(args.metadata_path, "r") as f):
        try:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
            args.dataset_name = config_data[0].get('name')
            args.target_attribute = config_data[0].get('dataset').get('target')
            args.task_type = config_data[0].get('dataset').get('type')
            try:
                args.data_path = f"{args.dataset_path}/{args.dataset_name}/{args.dataset_name}_{args.source_data}_train.csv"
                args.out_path = f"{args.output_dir}/{args.dataset_name}/{args.dataset_name}_aug_{args.source_data}_train.csv"
            except Exception as ex:
                raise Exception(ex)

        except yaml.YAMLError as ex:
            raise Exception(ex)

    return args


if __name__ == '__main__':
    args = parse_arguments()
    print(f"Augmentation Started for Dataset {args.dataset_name} on {args.source_data}:")

    if Path(args.data_path).is_file():
        df = pd.read_csv(args.data_path, na_values=[' ', '?', '-'], low_memory=False, encoding="ISO-8859-1")
        pd = augmentation(data=df, target_attribute=args.target_attribute, task_type=args.task_type)
        pd.to_csv(args.out_path, index=False)

        config_strs = [f"- name: {args.dataset_name}",
                       "  dataset:",
                       f"    train: \'{args.dataset_name}/{args.dataset_name}_aug_{args.source_data}_train.csv\'",
                       f"    test: \'{args.dataset_name}/{args.dataset_name}_test.csv\'",
                       f"    target_table: {args.dataset_name}",
                       f"    target: '{args.target_attribute}'",
                       f"    type: {args.task_type}"
                       "\n"]
        config_str = "\n".join(config_strs)

        yaml_file_local = f'{args.dataset_path}/{args.dataset_name}/{args.dataset_name}_aug_{args.source_data}.yaml'
        f_local = open(yaml_file_local, 'w')
        f_local.write("--- \n \n")
        f_local.write(config_str)
        f_local.close()
    else:
        print(f"Dataset is not exist: {args.data_path}")