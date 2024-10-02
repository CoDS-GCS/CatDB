from argparse import ArgumentParser
from util.FileHandler import read_text_file_line_by_line
from runcode.RunLocalPipeLine import run_local_pipeline


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--root-data-path', type=str, default=None)
    parser.add_argument('--dataset-name', type=str, default=None)
    parser.add_argument('--dataset-train', type=str, default=None)
    parser.add_argument('--dataset-test', type=str, default=None)
    parser.add_argument('--task-type', type=str, default=None)
    parser.add_argument('--dataset-description', type=str, default="yes")
    parser.add_argument('--prompt-representation-type', type=str, default=None)
    parser.add_argument('--prompt-samples-type', type=str, default=None)
    parser.add_argument('--prompt-number-samples', type=int, default=None)
    parser.add_argument('--prompt-number-iteration', type=int, default=1)
    parser.add_argument('--pipeline-path', type=str, default=None)
    parser.add_argument('--llm-model', type=str, default=None)
    parser.add_argument('--result-output-path', type=str, default="/tmp/results.csv")

    args = parser.parse_args()

    if args.root_data_path is None:
        raise Exception("--root-data-path is a required parameter!")

    if args.prompt_samples_type is None:
        raise Exception("--prompt-sample-type is a required parameter!")

    if args.prompt_number_samples is None:
        raise Exception("--prompt-number-samples is a required parameter!")

    if args.llm_model is None:
        raise Exception("--llm-model is a required parameter!")

    if args.prompt_number_iteration is None:
        args.prompt_number_iteration = 1

    args.data_source_train_path = f"{args.root_data_path}/{args.dataset_name}/{args.dataset_train}.csv"
    args.data_source_test_path = f"{args.root_data_path}/{args.dataset_name}/{args.dataset_test}.csv"


    if args.dataset_description.lower() == "yes":
        dataset_description_path = args.metadata_path.replace(".yaml", ".txt")
        args.description = read_text_file_line_by_line(fname=dataset_description_path)
        args.dataset_description = 'Yes'
    else:
        args.description = None
        args.dataset_description = 'No'

    return args


if __name__ == '__main__':
    from util.Config import __execute_mode

    args = parse_arguments()
    class_name = f"{args.prompt_representation_type}-{args.prompt_samples_type}-{args.prompt_number_samples}-SHOT"
    src_file_name = f"{args.llm_model}-{class_name}-{args.dataset_description}-iteration-{args.prompt_number_iteration}"
    file_name = f'{args.pipeline_path}/{src_file_name}-RUN.py'
    run_local_pipeline(args=args, file_name=file_name, run_mode=__execute_mode)