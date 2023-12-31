from argparse import ArgumentParser
from .catalog.Catalog import load_data_source_profile
from .prompt.PromptBuilder import prompt_factory


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--data-source-path', type=str, default=None)
    parser.add_argument('--data-source-name', type=str, default=None)
    parser.add_argument('--prompt-representation-type', type=str, default=None)
    parser.add_argument('--prompt-example-type', type=str, default=None)
    parser.add_argument('--prompt-number-example', type=str, default=None)
    parser.add_argument('--prompt-number-iteration', type=str, default=None)
    parser.add_argument('--task-type', type=str, default=None)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--target-attribute', type=str, default=None)
    args = parser.parse_args()

    if args.data_source_path is None:
        raise Exception("--data-source-path is a required parameter!")

    if args.data_source_name is None:
        raise Exception("--data-source-name is a required parameter!")

    if args.prompt_representation_type is None:
        raise Exception("--prompt-representation-type is a required parameter!")

    if args.prompt_example_type is None:
        raise Exception("--prompt-example-type is a required parameter!")

    if args.prompt_number_example is None:
        raise Exception("--prompt-number-example is a required parameter!")

    if args.task_type is None:
        raise Exception("--task-type is a required parameter!")

    if args.target_attribute is None:
        raise Exception("--target-attribute is a required parameter!")

    if args.prompt_number_iteration is None:
        args.prompt_number_iteration = 1

    return args


if __name__ == '__main__':
    args = parse_arguments()

    profile_info_path = f'{args.data_source_path}/{args.data_source_name}/'
    catalog = load_data_source_profile(data_source_path=profile_info_path, file_format="JSON")

    prompt = prompt_factory(catalog=catalog,
                            representation_type=args.promt_representation_type,
                            example_type=args.prompt_example_type,
                            number_example=args.prompt_number_example,
                            task_type=args.task_type,
                            number_iteration=args.prompt_number_iteration,
                            target_attribute=args.target_attribute)

    prompt_text = prompt.format(example=None)
