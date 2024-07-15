from argparse import ArgumentParser
import yaml
import os

from util.Config import Config
from util.Data import Dataset, reader_CSV


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--max-runtime-seconds', type=int, default=None)
    parser.add_argument('--exe-runtime-path', type=str, default=None)
    parser.add_argument('--iteration', type=int, default=1)
    parser.add_argument('--jvm-memory', type=int, default=2 * 1024)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--dataset-path', type=str, default=None)
    parser.add_argument('--llm-model', type=str, default=None)
    args = parser.parse_args()

    if args.metadata_path is None:
        raise Exception("--metadata-path is a required parameter!")

    if args.jvm_memory is not None:
        args.jvm_memory = args.jvm_memory * 1024

    # read .yaml file and extract values:
    with open(args.metadata_path, "r") as f:
        try:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
            args.dataset_name = config_data[0].get('name')
            args.target_attribute = config_data[0].get('dataset').get('target')
            args.task_type = config_data[0].get('dataset').get('type')
            try:
                args.data_source_train_path = args.dataset_path+ "/" + config_data[0].get('dataset').get('train').replace("{user}/", "")
                args.data_source_test_path = args.dataset_path + "/"+ config_data[0].get('dataset').get('test').replace("{user}/","")
            except Exception as ex:
                raise Exception(ex)

        except yaml.YAMLError as ex:
            raise Exception(ex)

    if args.max_runtime_seconds is None or args.exe_runtime_path is not None:
        try:
            df_exe = reader_CSV(args.exe_runtime_path)
            df_exe = df_exe.loc[(df_exe['dataset_name'] == args.dataset_name) &
                                (df_exe['llm_model'] == args.llm_model)]
            exe_time = df_exe["time"].values[0] - df_exe['dataset_load_time'].values[0]
            if exe_time < 1:
                exe_time = 1
            args.max_runtime_seconds = int(exe_time)
        except Exception as ex:
            raise Exception("--max-runtime-seconds is a required parameter!")

    args.config = Config(jvm_memory=args.jvm_memory,
                    max_runtime_seconds=args.max_runtime_seconds,
                    nthreads= os.cpu_count(),
                    output_predictions_file_train=f"{args.output_path}/{args.dataset_name}/train",
                    output_predictions_file_test=f"{args.output_path}/{args.dataset_name}/test",
                    output_dir=f"{args.output_dir}/{args.dataset_name}/",
                    output_path=args.output_path, llm_model=args.llm_model)

    args.dataset = Dataset(dataset_name=args.dataset_name,
                      train_path=args.data_source_train_path,
                      test_path=args.data_source_test_path,
                      task_type=args.task_type,
                      target_attribute=args.target_attribute)
    return args