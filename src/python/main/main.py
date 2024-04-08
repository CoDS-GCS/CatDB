from argparse import ArgumentParser
from catalog.Catalog import load_data_source_profile, CatalogInfo
from prompt.PromptBuilder import prompt_factory
from llm.GenerateLLMCode import GenerateLLMCode
from runcode.RunCode import RunCode
from util.FileHandler import save_prompt
from util.FileHandler import save_text_file, read_text_file_line_by_line
from pipegen.Metadata import Metadata
import pandas as pd
import time
import yaml


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--data-profile-path', type=str, default=None)
    parser.add_argument('--data-description', type=bool, default=False)
    parser.add_argument('--prompt-representation-type', type=str, default=None)
    parser.add_argument('--prompt-example-type', type=str, default=None)
    parser.add_argument('--prompt-number-example', type=int, default=None)
    parser.add_argument('--prompt-number-iteration', type=int, default=1)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--llm-model', type=str, default=None)
    parser.add_argument('--parse-pipeline', type=bool, default=True)
    parser.add_argument('--run-pipeline', type=bool, default=True)
    parser.add_argument('--enable-reduction', type=bool, default=False)
    parser.add_argument('--result-output-path', type=str, default="/tmp/results.csv")

    args = parser.parse_args()

    if args.metadata_path is None:
        raise Exception("--metadata-path is a required parameter!")

    if args.data_profile_path is None:
        raise Exception("--data-profile-path is a required parameter!")

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

    if args.prompt_representation_type == "CatDB":
        args.enable_reduction = True

    if args.description_path:
        dataset_description_path = args.metadata_path.replace(".yml",".txt")
        args.dataset_description = read_text_file_line_by_line(fname=dataset_description_path)
    else:
        args.dataset_description = None
    return args


def get_error_prompt(pipeline_code: str, pipeline_error: str):
    prompt_rule = ['You are expert in coding assistant. Your task is fix the error of this pipeline code.\n'
                   'The user will provide a pipeline code enclosed in "<CODE> pipline code will be here. </CODE>", '
                   'and an error message enclosed in "<ERROR> error message will be here. </ERROR>".',
                   'Fix the code error provided and return only the corrected pipeline without additional explanations'
                   ' regarding the resolved error.']
    prompt_msg = ["<CODE>\n",
                  pipeline_code,
                  "</CODE>\n",
                  "\n",
                  "<ERROR>\n",
                  pipeline_error,
                  "</ERROR>\n",
                  "Question: Fix the code error provided and return only the corrected pipeline without additional\n"
                  " explanations regarding the resolved error.\n"]

    return "".join(prompt_rule), "".join(prompt_msg)


def generate_and_run_pipeline(catalog: CatalogInfo, prompt_representation_type: str, args):
    # time
    gen_time = 0
    execute_time = 0
    final_status = False

    # Start Time
    start = time.time()
    prompt = prompt_factory(catalog=catalog,
                            representation_type=prompt_representation_type,
                            example_type=args.prompt_example_type,
                            number_example=args.prompt_number_example,
                            task_type=args.task_type,
                            number_iteration=args.prompt_number_iteration,
                            target_attribute=args.target_attribute,
                            data_source_train_path=args.data_source_train_path,
                            data_source_test_path=args.data_source_test_path,
                            number_folds=args.number_folds,
                            dataset_description=args.dataset_description)

    end = time.time()
    gen_time += end - start

    # Generate LLM code
    start = time.time()
    llm = GenerateLLMCode(model=args.llm_model)
    prompt_format = prompt.format(examples=None)
    prompt_rule = prompt_format["rules"]
    prompt_msg = prompt_format["question"]
    code = llm.generate_llm_code(prompt_rules=prompt_rule, prompt_message=prompt_msg)
    end = time.time()

    for i in range(5):
        if code == "Insufficient information.":
            start = time.time()
            code = llm.generate_llm_code(prompt_rules=prompt_rule, prompt_message=prompt_msg)
            end = time.time()
        else:
            break

    gen_time += end - start

    file_name = f'{args.output_path}/{args.llm_model}-{prompt.class_name}'

    # Save prompt:
    prompt_fname = f"{file_name}.prompt"
    save_prompt(fname=prompt_fname, prompt_rule=prompt_rule, prompt_msg=prompt_msg)

    # Parse LLM Code
    result = {"Train_Accuracy": -1, "Train_F1_score": -1, "Train_Log_loss": -1, "Train_R_Squared": -1, "Train_RMSE": -1,
              "Test_Accuracy": -1, "Test_F1_score": -1, "Test_Log_loss": -1, "Test_R_Squared": -1, "Test_RMSE": -1}
    iteration = 0
    if args.parse_pipeline or args.run_pipeline:
        rc = RunCode()
        parse = None
        for i in range(iteration, args.prompt_number_iteration):

            pipeline_fname = f"{file_name}_draft.py"
            save_text_file(fname=pipeline_fname, data=code)

            start = time.time()
            result = rc.execute_code(src=code, parse=parse)
            end = time.time()
            save_text_file(fname=pipeline_fname, data=code)
            if result.get_status():
                pipeline_fname = f"{file_name}.py"
                save_text_file(fname=pipeline_fname, data=code)

                execute_time = end - start
                final_status = True
                iteration = i + 1
                break
            else:
                error_fname = f"{file_name}_{i}.error"
                pipeline_fname = f"{file_name}_{i}.python"

                save_text_file(error_fname, f"{result.get_exception()}")
                save_text_file(fname=pipeline_fname, data=code)

                prompt_rule, prompt_msg = get_error_prompt(code, f"{result.get_exception()}")
                code = llm.generate_llm_code(prompt_rules=prompt_rule, prompt_message=prompt_msg)

    return final_status, iteration, gen_time, execute_time, result.parse_results()


if __name__ == '__main__':
    args = parse_arguments()

    start = time.time()
    catalog = load_data_source_profile(data_source_path=args.data_profile_path,
                                       file_format="JSON",
                                       target_attribute=args.target_attribute,
                                       enable_reduction=args.enable_reduction)

    end = time.time()
    catalog_time = end - start
    if args.prompt_representation_type == "AUTO":
        combinations = Metadata(catalog=catalog).get_combinations()
    else:
        combinations = [args.prompt_representation_type]

    try:
        df_result = pd.read_csv(args.result_output_path)
    except Exception as err:
        df_result = pd.DataFrame(columns=["dataset_name", "config", "task_type", "with_dataset_description", "status", "number_iteration", "pipeline_gen_time",
                                      "execution_time",
                                      "train_accuracy", "train_f1_score", "train_log_loss", "train_r_squared", "train_rmse",
                                      "test_accuracy", "test_f1_score", "test_log_loss", "test_r_squared", "test_rmse"])
    for rep_type in combinations:
        try:
            status, number_iteration, gen_time, execute_time, result = generate_and_run_pipeline(catalog=catalog,
                                                                                                 prompt_representation_type=rep_type,
                                                                                                 args=args)
            df_result.loc[len(df_result)] = [args.dataset_name, rep_type, args.task_type, args.dataset_description, status, number_iteration,
                                             catalog_time + gen_time, execute_time,
                                             result["Train_Accuracy"], result["Train_F1_score"], result["Train_Log_loss"],
                                             result["Train_R_Squared"], result["Train_RMSE"],
                                             result["Test_Accuracy"], result["Test_F1_score"],
                                             result["Test_Log_loss"],
                                             result["Test_R_Squared"], result["Test_RMSE"]
                                             ]

        except Exception as err:
            print("*******************************************")
            print(args.dataset_name)
            print(err)
    df_result.to_csv(args.result_output_path, index=False)
