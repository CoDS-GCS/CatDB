from argparse import ArgumentParser
from llm.GenerateLLMCode import GenerateLLMCode
from runcode.RunCode import RunCode
from util.FileHandler import save_text_file, read_text_file_line_by_line
import pandas as pd
import time
import yaml


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--src-path', type=str, default=None)
    parser.add_argument('--dataset-description', type=str, default="yes")
    parser.add_argument('--prompt-representation-type', type=str, default=None)
    parser.add_argument('--prompt-example-type', type=str, default=None)
    parser.add_argument('--prompt-number-example', type=int, default=None)
    parser.add_argument('--prompt-number-iteration', type=int, default=1)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--llm-model', type=str, default=None)
    parser.add_argument('--parse-pipeline', type=bool, default=True)
    parser.add_argument('--run-pipeline', type=bool, default=True)
    parser.add_argument('--result-output-path', type=str, default="/tmp/results.csv")

    args = parser.parse_args()

    if args.metadata_path is None:
        raise Exception("--metadata-path is a required parameter!")

    if args.src_path is None:
        raise Exception("--src-path is a required parameter!")

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

    if args.dataset_description.lower() == "yes":
        dataset_description_path = args.metadata_path.replace(".yaml", ".txt")
        args.description = read_text_file_line_by_line(fname=dataset_description_path)
        args.dataset_description = 'Yes'
    else:
        args.description = None
        args.dataset_description = 'No'
    return args


def get_error_prompt(pipeline_code: str, pipeline_error: str):
    min_length = min(len(pipeline_error), 2000)
    small_error_msg = pipeline_error[:min_length]
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
                  small_error_msg,
                  "</ERROR>\n",
                  "Question: Fix the code error provided and return only the corrected pipeline without additional\n"
                  " explanations regarding the resolved error.\n"]

    return "".join(prompt_rule), "".join(prompt_msg)


def run_pipeline(src: str, args):
    execute_time = 0
    final_status = False

    class_name = f"{args.prompt_representation_type}-{args.prompt_example_type}-{args.prompt_number_example}-SHOT"
    file_name = f'{args.output_path}/{args.llm_model}-{class_name}-{args.dataset_description}'

    # Parse LLM Code
    result = {"Train_Accuracy": -1, "Train_F1_score": -1, "Train_Log_loss": -1, "Train_R_Squared": -1, "Train_RMSE": -1,
              "Test_Accuracy": -1, "Test_F1_score": -1, "Test_Log_loss": -1, "Test_R_Squared": -1, "Test_RMSE": -1}
    iteration = 0
    code = src
    llm = GenerateLLMCode(model=args.llm_model)
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

    return final_status, iteration, 0, execute_time, result.parse_results()


if __name__ == '__main__':
    args = parse_arguments()

    try:
        df_result = pd.read_csv(args.result_output_path)

    except Exception as err:
        df_result = pd.DataFrame(columns=["dataset_name",
                                          "config",
                                          "llm_model",
                                          "has_description",
                                          "classifier",
                                          "task_type",
                                          "status",
                                          "number_iteration",
                                          "pipeline_gen_time",
                                          "execution_time",
                                          "train_accuracy",
                                          "train_f1_score",
                                          "train_log_loss",
                                          "train_r_squared",
                                          "train_rmse",
                                          "test_accuracy",
                                          "test_f1_score",
                                          "test_log_loss",
                                          "test_r_squared",
                                          "test_rmse"])

    try:
       src = read_text_file_line_by_line(args.src_path)
       status, number_iteration, gen_time, execute_time, result = run_pipeline(src=src, args=args)
       df_result.loc[len(df_result)] = [args.dataset_name,
                                             args.prompt_representation_type,
                                             args.llm_model,
                                             args.dataset_description,
                                             "automatic",
                                             args.task_type,
                                             status,
                                             number_iteration,
                                             0,
                                             execute_time,
                                             result["Train_Accuracy"],
                                             result["Train_F1_score"],
                                             result["Train_Log_loss"],
                                             result["Train_R_Squared"],
                                             result["Train_RMSE"],
                                             result["Test_Accuracy"],
                                             result["Test_F1_score"],
                                             result["Test_Log_loss"],
                                             result["Test_R_Squared"],
                                             result["Test_RMSE"]
                                             ]

    except Exception as err:
      print("*******************************************")
      print(args.dataset_name)
      print(err)

    df_result.to_csv(args.result_output_path, index=False)
