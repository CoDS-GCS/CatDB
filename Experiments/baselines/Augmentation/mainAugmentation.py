from argparse import ArgumentParser
import yaml
from Augmentation import augmentation
import pandas as pd

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--dataset-path', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
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
                args.data_path = f"{args.dataset_path}/{args.dataset_name}/{args.dataset_name}_orig_train.csv"
                args.out_path = f"{args.output_dir}/{args.dataset_name}/{args.dataset_name}_aug_train.csv"
            except Exception as ex:
                raise Exception(ex)

        except yaml.YAMLError as ex:
            raise Exception(ex)

    return args


if __name__ == '__main__':
    args = parse_arguments()
    print(f"Augmentation Started for Dataset {args.dataset_name}:")
    df = pd.read_csv(args.data_path, na_values=[' ', '?', '-'], low_memory=False, encoding="ISO-8859-1")
    pd = augmentation(data=df, target_attribute=args.target_attribute, task_type=args.task_type)
    pd.to_csv(args.out_path, index=False)