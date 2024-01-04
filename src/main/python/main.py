from argparse import ArgumentParser
from catalog.Catalog import load_data_source_profile
from prompt.PromptBuilder import prompt_factory
from llm.GenerateLLMCode import generate_llm_code
import yaml


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--data-source-path', type=str, default=None)
    parser.add_argument('--data-source-name', type=str, default=None)
    parser.add_argument('--prompt-representation-type', type=str, default=None)
    parser.add_argument('--prompt-example-type', type=str, default=None)
    parser.add_argument('--prompt-number-example', type=int, default=None)
    parser.add_argument('--prompt-number-iteration', type=int, default=None)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--llm-model', type=str, default=None)
    args = parser.parse_args()

    if args.data_source_path is None:
        raise Exception("--data-source-path is a required parameter!")

    if args.data_source_name is None:
        raise Exception("--data-source-name is a required parameter!")

    # read .yaml file and extract values:
    with open(f"{args.data_source_path}/{args.data_source_name}/{args.data_source_name}.yaml", "r") as f:
        try:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
            args.target_attribute = config_data[0].get('dataset').get('target')
            args.task_type = config_data[0].get('dataset').get('type')

            # TODO: read train and test dataset path from yaml file
            args.data_source_train_path = f"data/{args.data_source_name}/{args.data_source_name}_train.csv"
            args.data_source_test_path = f"data/{args.data_source_name}/{args.data_source_name}_test.csv"
        except yaml.YAMLError as exc:
            raise Exception(exc)

    if args.prompt_representation_type is None:
        raise Exception("--prompt-representation-type is a required parameter!")

    if args.prompt_example_type is None:
        raise Exception("--prompt-example-type is a required parameter!")

    if args.prompt_number_example is None:
        raise Exception("--prompt-number-example is a required parameter!")

    if args.llm_model is None:
        raise Exception("--llm-model is a required parameter!")

    if args.prompt_number_iteration is None:
        args.prompt_number_iteration = 1

    return args


if __name__ == '__main__':
    args = parse_arguments()
    profile_info_path = f'{args.data_source_path}/{args.data_source_name}/data_profile'
    catalog = load_data_source_profile(data_source_path=profile_info_path, file_format="JSON")

    prompt = prompt_factory(catalog=catalog,
                            representation_type=args.prompt_representation_type,
                            example_type=args.prompt_example_type,
                            number_example=args.prompt_number_example,
                            task_type=args.task_type,
                            number_iteration=args.prompt_number_iteration,
                            target_attribute=args.target_attribute,
                            data_source_train_path=args.data_source_train_path,
                            data_source_test_path=args.data_source_test_path)

    prompt_text = prompt.format(example=None)

    # Generate LLM code
    code = generate_llm_code(model=args.llm_model, prompt=prompt_text)

    # Save prompt text
    if args.output_path is not None:
        prompt_file_name = f'{args.output_path}/{args.data_source_name}-{prompt.class_name}-{args.llm_model}'
        f = open(f"{prompt_file_name}.txt", 'w')
        f.write(prompt_text)
        f.close()

        f = open(f"{prompt_file_name}.py", 'w')
        f.write(code)
        f.close()
