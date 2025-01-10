from argparse import ArgumentParser
from PrepareData import PrepareData
import yaml


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--dataset-path', type=str, default=None)
    parser.add_argument('--result-output-path', type=str, default=None)

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
                args.train_data_path = args.dataset_path + "/" + config_data[0].get('dataset').get('train').replace("{user}/", "")
                args.test_data_path = args.dataset_path + "/" + config_data[0].get('dataset').get('test').replace("{user}/", "")
            except Exception as ex:
                raise Exception(ex)

        except yaml.YAMLError as ex:
            raise Exception(ex)

    return args


if __name__ == '__main__':
    args = parse_arguments()

    pd = PrepareData(args.dataset_name, args.target_attribute, args.task_type, args.train_data_path,
                     args.test_data_path, args.output_dir, args.result_output_path)
    pd.run()
