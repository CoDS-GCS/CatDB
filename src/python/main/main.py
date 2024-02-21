from argparse import ArgumentParser
from catalog.Catalog import load_data_source_profile
from prompt.PromptBuilder import prompt_factory
from llm.GenerateLLMCode import GenerateLLMCode
from runcode.RunCode import RunCode
from runcode.CodeResultTemplate import CodeResultTemplate
from util.FileHandler import save_text_file
import yaml


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--data-profile-path', type=str, default=None)
    parser.add_argument('--prompt-representation-type', type=str, default=None)
    parser.add_argument('--prompt-example-type', type=str, default=None)
    parser.add_argument('--prompt-number-example', type=int, default=None)
    parser.add_argument('--prompt-number-iteration', type=int, default=1)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--llm-model', type=str, default=None)
    parser.add_argument('--parse-pipeline', type=bool, default=True)
    parser.add_argument('--run-pipeline', type=bool, default=True)

    args = parser.parse_args()

    if args.metadata_path is None:
        raise Exception("--metadata-path is a required parameter!")

    if args.data_profile_path is None:
        raise Exception("--data-profile-path is a required parameter!")

    # read .yaml file and extract values:
    with open(args.metadata_path, "r") as f:
        try:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
            args.target_attribute = config_data[0].get('dataset').get('target')
            args.task_type = config_data[0].get('dataset').get('type')
            try:
                args.data_source_train_path = config_data[0].get('dataset').get('train').replace("{user}/","")
                args.data_source_test_path = config_data[0].get('dataset').get('test').replace("{user}/", "")
            except Exception as ex:
                raise Exception(ex)

            try:
                args.number_folds = int(config_data[0].get('folds'))
            except yaml.YAMLError as ex:
                args.number_folds = 1

        except yaml.YAMLError as ex:
            raise Exception(ex)

    if args.prompt_example_type is None:
        raise Exception("--prompt-example-type is a required parameter!")

    if args.prompt_number_example is None:
        raise Exception("--prompt-number-example is a required parameter!")

    if args.llm_model is None:
        raise Exception("--llm-model is a required parameter!")

    if args.prompt_number_iteration is None:
        args.prompt_number_iteration = 1

    if args.prompt_representation_type is None:
        args.prompt_representation_type = "AUTO"

    return args


if __name__ == '__main__':
    args = parse_arguments()
    catalog = load_data_source_profile(data_source_path=args.data_profile_path,
                                               file_format="JSON",
                                               target_attribute=args.target_attribute)

    prompt = prompt_factory(catalog=catalog,
                            representation_type=args.prompt_representation_type,
                            example_type=args.prompt_example_type,
                            number_example=args.prompt_number_example,
                            task_type=args.task_type,
                            number_iteration=args.prompt_number_iteration,
                            target_attribute=args.target_attribute,
                            data_source_train_path=args.data_source_train_path,
                            data_source_test_path=args.data_source_test_path,
                            number_folds = args.number_folds)

    # Generate LLM code
    #llm = GenerateLLMCode(model=args.llm_model)
    prompt_format = prompt.format(examples=None)
    prompt_rule = prompt_format["rules"]
    prompt_msg = prompt_format["question"]
    #code = llm.generate_llm_code( prompt_rules=prompt_rule, prompt_message=prompt_msg)
    code = ""

    file_name = f'{args.output_path}/{args.llm_model}-{prompt.class_name}'
    # Save prompt:
    prompt_items = ["SYSTEM MESSAGE:\n",
                    prompt_rule,
                    "\n---------------------------------------\n",
                    "PROMPT TEXT:\n",
                    prompt_msg]
    prompt_data = "".join(prompt_items)
    prompt_fname = f"{file_name}.prompt"
    save_text_file(fname = prompt_fname, data = prompt_data)

    code_fname = f"{file_name}.py"
    save_text_file(fname=code_fname, data=code)

    # # Parse LLM Code
    # if args.parse_pipeline or args.run_pipeline:
    #     rc = RunCode()
    #     parse = None
    #     for i in range(1, args.prompt_number_iteration):
    #         result = rc.parse_code(code)
    #         if result.get_status() :
    #             pipeline_fname = f"{file_name}.py"
    #             save_text_file(fname=pipeline_fname, data=code)
    #             break
    #         else:

