from argparse import ArgumentParser
import time
import datetime
import yaml
import os
import numpy as np

from util.Config import Config
from util.Data import Dataset
from automl.H2O import H2O

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--result-output-path', type=str, default="/tmp/results.csv")
    parser.add_argument('--error-output-path', type=str, default="/tmp/catdb_error.csv")
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
                args.data_source_train_path = "../../../" + config_data[0].get('dataset').get('train').replace(
                    "{user}/", "")
                args.data_source_test_path = "../../../" + config_data[0].get('dataset').get('test').replace("{user}/",
                                                                                                             "")
            except Exception as ex:
                raise Exception(ex)

        except yaml.YAMLError as ex:
            raise Exception(ex)

    return args


def call_run(result, path):
    try:
        res = dict(result)
        for name in ['predictions', 'truth', 'probabilities']:
            arr = result[name]
            if arr is not None:
                np.savetxt(f"{path}-{name}.csv", arr, delimiter=",", fmt='%s')
    except BaseException as e:
        print(str(e))
        res = dict(
            error_message=str(e),
            models_count=0
        )


if __name__ == '__main__':
    config = Config(jvm_memory=10240, max_runtime_seconds=30, nthreads=12,
                    output_predictions_file_train="/home/saeed/Documents/Adult/adoutl_train",
                    output_predictions_file_test="/home/saeed/Documents/Adult/adoutl_test",
                    output_dir="/home/saeed/Documents/Adult")

    dataset = Dataset("Adult", train_path="/home/saeed/Documents/Github/CatDB/Experiments/data/Adult/Adult_train.csv",
                      test_path="/home/saeed/Documents/Github/CatDB/Experiments/data/Adult/Adult_test.csv",
                      task_type="binary", target_attribute="class")

    ml = H2O(dataset=dataset, config=config)
    results = ml.run()
    call_run(result=results["train_result"], path=config.output_predictions_file_train)