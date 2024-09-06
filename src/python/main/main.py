from argparse import ArgumentParser
from catalog.Catalog import load_data_source_profile
from catalog.Dependency import load_dependency_info
from prompt.PromptBuilder import prompt_factory, error_prompt_factory, result_error_prompt_factory, prompt_factory_data_cleaning
from llm.GenerateLLMCode import GenerateLLMCode
from runcode.RunCode import RunCode
from util.FileHandler import save_prompt
from util.FileHandler import save_text_file, read_text_file_line_by_line
from util.Config import load_config
from util.LogResults import save_log
from util.ErrorResults import ErrorResults
from pipegen.Metadata import Metadata
import time
import datetime
import yaml
import os


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--root-data-path', type=str, default=None)
    parser.add_argument('--catalog-path', type=str, default=None)
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
    parser.add_argument('--error-output-path', type=str, default="/tmp/catdb_error.csv")
    parser.add_argument('--run-code', type=bool, default=False)
    parser.add_argument('--system-log', type=str, default="/tmp/catdb-system-log.dat")

    args = parser.parse_args()

    if args.metadata_path is None:
        raise Exception("--metadata-path is a required parameter!")

    if args.root_data_path is None:
        raise Exception("--root-data-path is a required parameter!")

    if args.catalog_path is None:
        raise Exception("--catalog-path is a required parameter!")

    # read .yaml file and extract values:
    with open(args.metadata_path, "r") as f:
        try:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
            args.dataset_name = config_data[0].get('name')
            args.target_attribute = config_data[0].get('dataset').get('target')
            args.task_type = config_data[0].get('dataset').get('type')
            args.multi_table = config_data[0].get('dataset').get('multi_table')
            args.target_table = config_data[0].get('dataset').get('target_table')
            if args.multi_table is None or args.multi_table not in {True, False}:
                args.multi_table = False

            try:
                args.data_source_train_path = f"{args.root_data_path}/{config_data[0].get('dataset').get('train')}"
                args.data_source_test_path = f"{args.root_data_path}/{config_data[0].get('dataset').get('test')}"
                args.data_source_verify_path = f"{args.root_data_path}/{config_data[0].get('dataset').get('verify')}"
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


def clean_up(args, prompt_file_name):
    file_names = []
    file_name = f'{args.output_path}/{prompt_file_name}'
    prompt_fname = f"{file_name}.prompt"
    file_names.append(prompt_fname)

    pipeline_fname = f"{file_name}_draft.py"
    file_names.append(pipeline_fname)
    for i in range(0, args.prompt_number_iteration_error):
        error_fname = f"{file_name}_{i}.error"
        pipeline_fname = f"{file_name}_{i}.python"
        prompt_fname_error = f"{file_name}_Error_{i}.prompt"
        prompt_results_fname_error = f"{file_name}_Error_Results_{i}.prompt"
        file_names.append(error_fname)
        file_names.append(pipeline_fname)
        file_names.append(prompt_fname_error)
        file_names.append(prompt_results_fname_error)

    for fn in file_names:
        try:
            os.remove(fn)
        except:
            pass


def generate_and_verify_pipeline(args, catalog, run_mode: str = None, sub_task: str = '', previous_result: str = None,
                              time_catalog: float = 0, iteration: int = 1, dependency: dict() = None):
    all_token_count = 0
    time_generate = 0
    time_execute = 0
    final_status = False
    time_total = 0

    # #---------------------------------------
    # data_cleaning_prompt = prompt_factory_data_cleaning(catalog=catalog)
    # prompt_format = data_cleaning_prompt.format()
    # prompt_system_message = prompt_format["system_message"]
    # prompt_user_message = prompt_format["user_message"]
    #
    #
    # # Save prompt:
    # prompt_file_name = f"{args.llm_model}-{data_cleaning_prompt.class_name}-{args.dataset_description}-iteration-{iteration}"
    # file_name = f'{args.output_path}/{prompt_file_name}'
    # prompt_fname = f"{file_name}.prompt"
    # save_prompt(fname=prompt_fname, system_message=prompt_system_message, user_message=prompt_user_message)
    # #---------------------------------------

    # time_start_1 = time.time()  # Start Time
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
                            previous_result=previous_result,
                            target_table=args.target_table,
                            dependency= dependency)
    #
    # time_end_1 = time.time()  # End time
    # time_generate += time_end_1 - time_start_1  # Add prompt construction time to pipeline generate time
    #
    # prompt_format = prompt.format()
    # prompt_system_message = prompt_format["system_message"]
    # prompt_user_message = prompt_format["user_message"]
    # schema_data = prompt_format["schema_data"]
    #
    # # Save prompt:
    # prompt_file_name = f"{args.llm_model}-{prompt.class_name}-{args.dataset_description}-iteration-{iteration}"
    # file_name = f'{args.output_path}/{prompt_file_name}'
    #
    # clean_up(args=args, prompt_file_name=prompt_file_name)
    #
    # prompt_fname = f"{file_name}.prompt"
    # save_prompt(fname=prompt_fname, system_message=prompt_system_message, user_message=prompt_user_message)
    #
    # # Generate LLM code
    # code, prompt_token_count, time_tmp_gen = GenerateLLMCode.generate_llm_code(user_message=prompt_user_message,
    #                                                                            system_message=prompt_system_message)
    # time_generate_extra = 0
    # for i in range(5):
    #     if code == "Insufficient information.":
    #         code, tokens_count, time_tmp_gen = GenerateLLMCode.generate_llm_code(user_message=prompt_user_message,
    #                                                                              system_message=prompt_system_message)
    #         all_token_count += tokens_count
    #         time_generate_extra += time_tmp_gen
    #     else:
    #         break
    # time_generate += time_tmp_gen
    #
    # iteration_error = 0
    # results_verified = False
    # results = None
    # final_pipeline_file_name = None
    # for i in range(0, args.prompt_number_iteration_error):
    #     # Replace Original Train Data with Verify Data
    #     code = code.replace(args.data_source_train_path, args.data_source_verify_path)
    #
    #     if len(code) > 500:
    #         pipeline_fname = f"{file_name}_draft.py"
    #         save_text_file(fname=pipeline_fname, data=code)
    #
    #     time_start_2 = time.time()
    #     result = RunCode.execute_code(src=code, parse=None, run_mode=run_mode)
    #     time_end_2 = time.time()
    #     if result.get_status():
    #         results_verified, results = result.parse_results()
    #         pipeline_fname = f"{file_name}.py"
    #         save_text_file(fname=pipeline_fname, data=code)
    #         if results_verified:
    #             time_execute = time_end_2 - time_start_2
    #             final_status = True
    #             iteration_error = i
    #             final_pipeline_file_name = file_name
    #             break
    #         else:
    #             system_message, user_message = result_error_prompt_factory(pipeline_code=code, task_type=args.task_type,
    #                                                                        data_source_train_path=args.data_source_train_path,
    #                                                                        data_source_test_path=args.data_source_test_path)
    #             prompt_fname_error = f"{file_name}_Error_Results_{i}.prompt"
    #             save_prompt(fname=prompt_fname_error, system_message=system_message, user_message=user_message)
    #             new_code, tokens_count, time_tmp_gen = GenerateLLMCode.generate_llm_code(system_message=system_message,
    #                                                                                      user_message=user_message)
    #             time_total += time_tmp_gen
    #             if len(new_code) > 500:
    #                 all_token_count += tokens_count
    #                 code = new_code
    #             else:
    #                 i -= 1
    #
    #     else:
    #         # add error to error lists:
    #         ErrorResults(error_class=result.get_error_class(), error_exception=result.error_exception,
    #                      error_type=result.get_error_type(), error_value=result.get_error_value(),
    #                      error_detail=result.get_error_detail(), dataset_name=args.dataset_name,
    #                      llm_model=args.llm_model,
    #                      config=args.prompt_representation_type, sub_task=sub_task,
    #                      file_name=f"{prompt_file_name}_{i}.python",
    #                      timestamp=datetime.datetime.utcnow().isoformat()).save_error(args.error_output_path)
    #
    #         error_fname = f"{file_name}_{i}.error"
    #         pipeline_fname = f"{file_name}_{i}.python"
    #         save_text_file(error_fname, f"{result.get_exception()}")
    #         save_text_file(fname=pipeline_fname, data=code)
    #
    #         system_message, user_message = error_prompt_factory(pipeline_code=code,
    #                                                             pipeline_error_class = result.get_error_class(),
    #                                                             pipeline_error_detail = result.get_error_detail(),
    #                                                             schema_data=schema_data,
    #                                                             task_type=args.task_type,
    #                                                             data_source_train_path=args.data_source_train_path,
    #                                                             data_source_test_path=args.data_source_test_path)
    #         prompt_fname_error = f"{file_name}_Error_{i}.prompt"
    #         save_prompt(fname=prompt_fname_error, system_message=system_message, user_message=user_message)
    #
    #         new_code, tokens_count, time_tmp_gen = GenerateLLMCode.generate_llm_code(system_message=system_message,
    #                                                                                  user_message=user_message)
    #         time_total += time_tmp_gen
    #         if len(new_code) > 500:
    #             all_token_count += tokens_count
    #             code = new_code
    #         else:
    #             i -= 1
    #
    # time_total = time_total+time_generate_extra+time_generate+time_execute
    # save_log(args=args, sub_task=sub_task, iteration=iteration, iteration_error=iteration_error, time_catalog=time_catalog,
    #          time_generate=time_generate, time_total=time_total, time_execute=time_execute,
    #          prompt_token_count=prompt_token_count, all_token_count=all_token_count, operation_tag='Gen-and-Verify-Pipeline',
    #          run_mode=run_mode, results_verified=results_verified, results=results, final_status=final_status)
    #
    # if run_mode == __execute_mode :
    #     final_status, code = run_pipeline(args=args, file_name=final_pipeline_file_name, code=code, schema_data=schema_data,
    #                  run_mode=__execute_mode, sub_task=sub_task, iteration=iteration, time_total=time_total,
    #                  time_catalog=time_catalog, time_generate=time_generate, all_token_count=all_token_count,
    #                 prompt_token_count=prompt_token_count)
    #
    # return final_status, code
    return True, "-----------"


def run_pipeline(args, file_name, code, schema_data, run_mode, sub_task: str = '', iteration: int = 1,
                 time_total: int = 0, time_catalog: float = 0, time_generate: int = 0, all_token_count: int = 0,
                 prompt_token_count: int = 0):
    time_execute = 0
    final_status = False

    # Run pipeline with original data
    code = code.replace(args.data_source_verify_path, args.data_source_train_path)

    iteration_error = 0
    results_verified = False
    results = None
    for i in range(0, args.prompt_number_iteration_error):
        time_start_1 = time.time()
        result = RunCode.execute_code(src=code, parse=None, run_mode=run_mode)
        time_end_1 = time.time()
        if result.get_status():
            results_verified, results = result.parse_results()
            pipeline_fname = f"{file_name}-RUN.py"
            save_text_file(fname=pipeline_fname, data=code)
            if results_verified:
                time_execute = time_end_1 - time_start_1
                final_status = True
                iteration_error = i
                break
            else:
                system_message, user_message = result_error_prompt_factory(pipeline_code=code, task_type=args.task_type,
                                                                           data_source_train_path=args.data_source_train_path,
                                                                           data_source_test_path=args.data_source_test_path)
                prompt_fname_error = f"{file_name}_Error_Results_{i}_RUN.prompt"
                save_prompt(fname=prompt_fname_error, system_message=system_message, user_message=user_message)
                new_code, tokens_count, time_tmp_gen = GenerateLLMCode.generate_llm_code(system_message=system_message,
                                                                                         user_message=user_message)
                time_total += time_tmp_gen
                if len(new_code) > 500:
                    all_token_count += tokens_count
                    code = new_code
                else:
                    i -= 1

        else:
            # add error to error lists:
            ErrorResults(error_class=result.get_error_class(), error_exception=result.error_exception,
                         error_type=result.get_error_type(), error_value=result.get_error_value(),
                         error_detail=result.get_error_detail(), dataset_name=args.dataset_name,
                         llm_model=args.llm_model,
                         config=args.prompt_representation_type, sub_task=sub_task,
                         file_name=f"{file_name}_{i}_RUN.python",
                         timestamp=datetime.datetime.utcnow().isoformat()).save_error(args.error_output_path)

            error_fname = f"{file_name}_{i}_RUN.error"
            pipeline_fname = f"{file_name}_{i}_RUN.python"
            save_text_file(error_fname, f"{result.get_exception()}")
            save_text_file(fname=pipeline_fname, data=code)

            system_message, user_message = error_prompt_factory(pipeline_code=code,
                                                                pipeline_error_class = result.get_error_class(),
                                                                pipeline_error_detail = result.get_error_detail(),
                                                                schema_data=schema_data,
                                                                task_type=args.task_type,
                                                                data_source_train_path=args.data_source_train_path,
                                                                data_source_test_path=args.data_source_test_path)
            prompt_fname_error = f"{file_name}_Error_{i}_RUN.prompt"
            save_prompt(fname=prompt_fname_error, system_message=system_message, user_message=user_message)

            new_code, tokens_count, time_tmp_gen = GenerateLLMCode.generate_llm_code(system_message=system_message,
                                                                                     user_message=user_message)
            time_total += time_tmp_gen
            if len(new_code) > 500:
                all_token_count += tokens_count
                code = new_code
            else:
                i -= 1

    save_log(args=args, sub_task=sub_task, iteration=iteration, iteration_error=iteration_error,
             time_catalog=time_catalog, time_generate=time_generate, time_total=time_total+time_execute, time_execute=time_execute,
             prompt_token_count=prompt_token_count, all_token_count=all_token_count, operation_tag='Run-Pipeline',
             run_mode=run_mode, results_verified=results_verified, results=results, final_status=final_status)

    return final_status, code


if __name__ == '__main__':
    from util.Config import __execute_mode, __gen_verify_mode, __sub_task_data_preprocessing, \
        __sub_task_feature_engineering, __sub_task_model_selection

    args = parse_arguments()

    if args.run_code == False:
        operation = generate_and_verify_pipeline
        begin_iteration = 1
        end_iteration = 1
    else:
        operation = run_pipeline
        begin_iteration = args.prompt_number_iteration
        end_iteration = 1

    data_profile_path = f"{args.catalog_path}/data_profile"
    catalog = []
    time_start = time.time()
    dependency_file = f"{args.catalog_path}/dependency.yaml"
    dependencies = load_dependency_info(dependency_file=dependency_file, datasource_name=args.dataset_name)

    if args.multi_table:
        load_config(system_log=args.system_log, llm_model=args.llm_model, rules_path="RulesMultiTable.yaml")
        for tbl in dependencies.keys():
            tbl_dp_path = f"{data_profile_path}/{tbl}"
            enable_reduction = False
            if tbl == args.target_table:
                enable_reduction = True
            cat = load_data_source_profile(data_source_path=tbl_dp_path,
                                           file_format="JSON",
                                           target_attribute=args.target_attribute,
                                           enable_reduction=enable_reduction,
                                           dependency=dependencies[tbl])
            cat.table_name = tbl
            catalog.append(cat)
    else:
        load_config(system_log=args.system_log, llm_model=args.llm_model, rules_path="Rules.yaml")
        catalog.append(load_data_source_profile(data_source_path=data_profile_path,
                                                file_format="JSON",
                                                target_attribute=args.target_attribute,
                                                enable_reduction=args.enable_reduction))

    time_end = time.time()
    time_catalog = time_end - time_start
    ti = 0
    t = args.prompt_number_iteration * 2

    prompt_representation_type_orig = args.prompt_representation_type
    while begin_iteration < args.prompt_number_iteration + end_iteration:
        if args.prompt_representation_type == "CatDBChain":
            final_status, code = operation(args=args, catalog=catalog, run_mode=__gen_verify_mode, sub_task=__sub_task_data_preprocessing, time_catalog=time_catalog, iteration=begin_iteration, dependency=dependencies)
            if final_status:
                final_status, code = operation(args=args, catalog=catalog, run_mode=__gen_verify_mode, sub_task=__sub_task_feature_engineering, previous_result=code, time_catalog=time_catalog, iteration=begin_iteration, dependency=dependencies)
                if final_status:
                    final_status, code = operation(args=args, catalog=catalog, run_mode=__execute_mode, sub_task=__sub_task_model_selection, previous_result=code, time_catalog=time_catalog, iteration=begin_iteration, dependency=dependencies)
                    if final_status:
                        begin_iteration += 1
        elif args.prompt_representation_type == "AUTO":
            combinations = Metadata(catalog=catalog[0]).get_combinations()
            for cmb in combinations:
                args.prompt_representation_type = cmb
                final_status, code = operation(args=args, catalog=catalog, run_mode=__execute_mode, time_catalog=time_catalog, iteration=begin_iteration)
                if final_status:
                    begin_iteration += 1
            args.prompt_representation_type = prompt_representation_type_orig
        else:
            final_status, code = operation(args=args, catalog=catalog, run_mode=__execute_mode, time_catalog=time_catalog, iteration=begin_iteration, dependency=dependencies)
            if final_status:
                begin_iteration += 1

        ti += 1
        if ti > t:
            break
