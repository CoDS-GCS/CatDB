from argparse import ArgumentParser
from catalog.Catalog import load_data_source_profile
from prompt.PromptBuilder import prompt_factory, error_prompt_factory
from llm.GenerateLLMCode import GenerateLLMCode
from runcode.RunCode import RunCode
from util.FileHandler import save_prompt
from util.FileHandler import save_text_file, read_text_file_line_by_line
from util.Config import set_config
from util.LogResults import LogResults
from pipegen.Metadata import Metadata
import time
import yaml


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--data-profile-path', type=str, default=None)
    parser.add_argument('--dataset-description', type=str, default="yes")
    parser.add_argument('--prompt-representation-type', type=str, default=None)
    parser.add_argument('--prompt-samples-type', type=str, default=None)
    parser.add_argument('--prompt-number-samples', type=int, default=None)
    parser.add_argument('--prompt-number-iteration', type=int, default=1)
    parser.add_argument('--prompt-number-iteration-error', type=int, default=1)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--llm-model', type=str, default=None)
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

        except yaml.YAMLError as ex:
            raise Exception(ex)

    if args.prompt_samples_type is None:
        raise Exception("--prompt-sample-type is a required parameter!")

    if args.prompt_number_samples is None:
        raise Exception("--prompt-number-samples is a required parameter!")

    if args.llm_model is None:
        raise Exception("--llm-model is a required parameter!")

    if args.prompt_number_iteration is None:
        args.prompt_number_iteration = 1

    if args.prompt_representation_type in {"CatDB", "CatDBChain"}:
        args.enable_reduction = True

    if args.dataset_description.lower() == "yes":
        dataset_description_path = args.metadata_path.replace(".yaml", ".txt")
        args.description = read_text_file_line_by_line(fname=dataset_description_path)
        args.dataset_description = 'Yes'
    else:
        args.description = None
        args.dataset_description = 'No'
    return args


def generate_and_run_pipeline(args, catalog, run_mode: str = None, sub_task: str = '', previous_result: str = None,
                              time_catalog: float = 0, iteration: int = 1):

    from util.Config import __gen_run_mode
    time_generate = 0
    time_execute = 0
    final_status = False

    time_total_start = time.time()
    time_start = time.time()  # Start Time
    prompt = prompt_factory(catalog=catalog,
                            representation_type=f"{args.prompt_representation_type}{sub_task}",
                            samples_type=args.prompt_samples_type,
                            number_samples=args.prompt_number_samples,
                            task_type=args.task_type,
                            number_iteration=args.prompt_number_iteration,
                            target_attribute=args.target_attribute,
                            data_source_train_path=args.data_source_train_path,
                            data_source_test_path=args.data_source_test_path,
                            dataset_description=args.description,
                            previous_result=previous_result)

    time_end = time.time()  # End time
    time_generate += time_end - time_start  # Add prompt construction time to pipeline generate time

    prompt_format = prompt.format()
    prompt_system_message = prompt_format["system_message"]
    prompt_user_message = prompt_format["user_message"]

    # Save prompt:
    file_name = f'{args.output_path}/{args.llm_model}-{prompt.class_name}-{args.dataset_description}-iteration-{iteration}'
    if sub_task != '':
        file_name = f"{file_name}-{sub_task}"

    prompt_fname = f"{file_name}.prompt"
    save_prompt(fname=prompt_fname, system_message=prompt_system_message, user_message=prompt_user_message)

    # Generate LLM code
    # time_start = time.time()
    # code = GenerateLLMCode.generate_llm_code(user_message=prompt_user_message, system_message=prompt_system_message)
    # time_end = time.time()
    #
    # for i in range(5):
    #     if code == "Insufficient information.":
    #         time_start = time.time()
    #         code = GenerateLLMCode.generate_llm_code(user_message=prompt_user_message,
    #                                                  system_message=prompt_system_message)
    #         time_end = time.time()
    #     else:
    #         break
    # time_generate += time_end - time_start

    # iteration_error = 0
    # for i in range(iteration_error, args.prompt_number_iteration_error):
    #     if len(code) > 500:
    #         pipeline_fname = f"{file_name}_draft.py"
    #         save_text_file(fname=pipeline_fname, data=code)
    #
    #     time_start = time.time()
    #     result = RunCode.execute_code(src=code, parse=None, run_mode=run_mode)
    #     time_end = time.time()
    #     if result.get_status():
    #         pipeline_fname = f"{file_name}.py"
    #         save_text_file(fname=pipeline_fname, data=code)
    #
    #         time_execute = time_end - time_start
    #         final_status = True
    #         iteration_error = i + 1
    #         break
    #     else:
    #         error_fname = f"{file_name}_{i}.error"
    #         pipeline_fname = f"{file_name}_{i}.python"
    #
    #         save_text_file(error_fname, f"{result.get_exception()}")
    #         save_text_file(fname=pipeline_fname, data=code)
    #
    #         system_message, user_message = error_prompt_factory(code, f"{result.get_exception()}")
    #         prompt_fname_error = f"{file_name}_Error_{i}.prompt"
    #         save_prompt(fname=prompt_fname_error, system_message=system_message, user_message=user_message)
    #
    #         new_code = GenerateLLMCode.generate_llm_code(system_message=system_message, user_message=user_message)
    #         if len(new_code) > 500:
    #             code = new_code
    #         else:
    #             i -= 1
    #
    # time_total_end = time.time()
    # time_total = time_total_end - time_total_start
    #
    # log_results = LogResults(dataset_name=args.dataset_name, config=args.prompt_representation_type, sub_task=sub_task,
    #                          llm_model=args.llm_model, classifier="Auto", task_type=args.task_type,
    #                          status=f"{final_status}", number_iteration=iteration, number_iteration_error=iteration_error,
    #                          has_description=args.dataset_description,
    #                          time_catalog_load=time_catalog, time_pipeline_generate=time_generate,
    #                          time_total=time_total,
    #                          time_execution=time_execute)
    # if run_mode == __gen_run_mode:
    #     results = result.parse_results()
    #     log_results.train_accuracy = results["Train_Accuracy"]
    #     log_results.train_f1_score = results["Train_F1_score"]
    #     log_results.train_log_loss = results["Train_Log_loss"]
    #     log_results.train_r_squared = results["Train_R_Squared"]
    #     log_results.train_rmse = results["Train_RMSE"]
    #     log_results.test_accuracy = results["Test_Accuracy"]
    #     log_results.test_f1_score = results["Test_F1_score"]
    #     log_results.test_log_loss = results["Test_Log_loss"]
    #     log_results.test_r_squared = results["Test_R_Squared"]
    #     log_results.test_rmse = results["Test_RMSE"]
    #
    # log_results.save_results(result_output_path=args.result_output_path)

    # return final_status, code
    return True, "CODE"


if __name__ == '__main__':
    from util.Config import __validation_run_mode, __gen_run_mode, __sub_task_data_preprocessing, \
        __sub_task_feature_engineering, __sub_task_model_selection

    args = parse_arguments()
    set_config(args.llm_model)

    time_total_start = time_start = time.time()
    catalog = load_data_source_profile(data_source_path=args.data_profile_path,
                                       file_format="JSON",
                                       target_attribute=args.target_attribute,
                                       enable_reduction=args.enable_reduction)

    time_end = time.time()
    time_catalog = time_end - time_start

    for i in range(0, args.prompt_number_iteration):
        if args.prompt_representation_type == "CatDBChain":
            final_status, code = generate_and_run_pipeline(args=args, catalog=catalog, run_mode=__validation_run_mode,
                                                           sub_task=__sub_task_data_preprocessing, time_catalog=time_catalog, iteration=i)
            if final_status:
                final_status, code = generate_and_run_pipeline(args=args, catalog=catalog,  run_mode=__validation_run_mode,
                                                               sub_task=__sub_task_feature_engineering,
                                                               previous_result=code, time_catalog=time_catalog, iteration=i)
                if final_status:
                    final_status, code = generate_and_run_pipeline(args=args,catalog=catalog, run_mode=__gen_run_mode,
                                                                   sub_task=__sub_task_model_selection,
                                                                   previous_result=code, time_catalog=time_catalog, iteration=i)
        elif args.prompt_representation_type == "AUTO":
            combinations = Metadata(catalog=catalog).get_combinations()
            for cmb in combinations:
                args.prompt_representation_type = cmb
                generate_and_run_pipeline(args=args, catalog=catalog, run_mode=__gen_run_mode,
                                          time_catalog=time_catalog, iteration=i)
        else:
            generate_and_run_pipeline(args=args, catalog=catalog, run_mode=__gen_run_mode,
                                      time_catalog=time_catalog, iteration=i)
