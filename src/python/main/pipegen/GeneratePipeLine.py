from catalog.Catalog import load_data_source_profile
from prompt.PromptBuilder import prompt_factory, error_prompt_factory, result_error_prompt_factory, \
    prompt_factory_data_cleaning, prompt_factory_data_catalog_cleaning
from llm.GenerateLLMCode import GenerateLLMCode
from llmdataprepare.DataPrepareLLM import DataPrepareLLM
from catalog.Profile import ProfileInfoUpdate
from runcode.RunCode import RunCode
from util.FileHandler import save_prompt
from util.FileHandler import save_text_file
from util.LogResults import save_log, save_cleaning_log
from util.ErrorResults import ErrorResults
from util.DatasetReader import split_clean_data_save
import time
import datetime
import os
import pandas as pd


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
    from util.Config import __execute_mode
    all_token_count = 0
    time_generate = 0
    time_execute = 0
    final_status = False
    time_total = 0

    time_start_1 = time.time()  # Start Time
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
                            dependency=dependency)

    time_end_1 = time.time()  # End time
    time_generate += time_end_1 - time_start_1  # Add prompt construction time to pipeline generate time

    prompt_format = prompt.format()
    prompt_system_message = prompt_format["system_message"]
    prompt_user_message = prompt_format["user_message"]
    schema_data = prompt_format["schema_data"]

    # Save prompt:
    prompt_file_name = f"{args.llm_model}-{prompt.class_name}-{args.dataset_description}-iteration-{iteration}"
    file_name = f'{args.output_path}/{prompt_file_name}'

    clean_up(args=args, prompt_file_name=prompt_file_name)

    prompt_fname = f"{file_name}.prompt"
    save_prompt(fname=prompt_fname, system_message=prompt_system_message, user_message=prompt_user_message)

    # Generate LLM code
    code, prompt_token_count, time_tmp_gen = GenerateLLMCode.generate_llm_code(user_message=prompt_user_message,
                                                                               system_message=prompt_system_message)
    time_generate_extra = 0
    for i in range(5):
        if code == "Insufficient information.":
            code, tokens_count, time_tmp_gen = GenerateLLMCode.generate_llm_code(user_message=prompt_user_message,
                                                                                 system_message=prompt_system_message)
            all_token_count += tokens_count
            time_generate_extra += time_tmp_gen
        else:
            break
    time_generate += time_tmp_gen

    iteration_error = 0
    results_verified = False
    results = None
    final_pipeline_file_name = None
    for i in range(0, args.prompt_number_iteration_error):
        # Replace Original Train Data with Verify Data
        code = code.replace(args.data_source_train_path, args.data_source_verify_path)

        if len(code) > 500:
            pipeline_fname = f"{file_name}_draft.py"
            save_text_file(fname=pipeline_fname, data=code)

        time_start_2 = time.time()
        result = RunCode.execute_code(src=code, parse=None, run_mode=run_mode)
        time_end_2 = time.time()
        if result.get_status():
            results_verified, results = result.parse_results()
            pipeline_fname = f"{file_name}.py"
            save_text_file(fname=pipeline_fname, data=code)
            if results_verified:
                time_execute = time_end_2 - time_start_2
                final_status = True
                iteration_error = i
                final_pipeline_file_name = file_name
                break
            else:
                system_message, user_message = result_error_prompt_factory(pipeline_code=code, task_type=args.task_type,
                                                                           data_source_train_path=args.data_source_train_path,
                                                                           data_source_test_path=args.data_source_test_path)
                prompt_fname_error = f"{file_name}_Error_Results_{i}.prompt"
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
                         file_name=f"{prompt_file_name}_{i}.python",
                         timestamp=datetime.datetime.utcnow().isoformat()).save_error(args.error_output_path)

            error_fname = f"{file_name}_{i}.error"
            pipeline_fname = f"{file_name}_{i}.python"
            save_text_file(error_fname, f"{result.get_exception()}")
            save_text_file(fname=pipeline_fname, data=code)

            system_message, user_message = error_prompt_factory(pipeline_code=code,
                                                                pipeline_error_class=result.get_error_class(),
                                                                pipeline_error_detail=result.get_error_detail(),
                                                                schema_data=schema_data,
                                                                task_type=args.task_type,
                                                                data_source_train_path=args.data_source_train_path,
                                                                data_source_test_path=args.data_source_test_path)
            prompt_fname_error = f"{file_name}_Error_{i}.prompt"
            save_prompt(fname=prompt_fname_error, system_message=system_message, user_message=user_message)

            new_code, tokens_count, time_tmp_gen = GenerateLLMCode.generate_llm_code(system_message=system_message,
                                                                                     user_message=user_message)
            time_total += time_tmp_gen
            if len(new_code) > 500:
                all_token_count += tokens_count
                code = new_code
            else:
                i -= 1

    time_total = time_total + time_generate_extra + time_generate + time_execute
    save_log(args=args, sub_task=sub_task, iteration=iteration, iteration_error=iteration_error,
             time_catalog=time_catalog,
             time_generate=time_generate, time_total=time_total, time_execute=time_execute,
             prompt_token_count=prompt_token_count, all_token_count=all_token_count,
             operation_tag='Gen-and-Verify-Pipeline',
             run_mode=run_mode, results_verified=results_verified, results=results, final_status=final_status)

    if run_mode == __execute_mode and final_status:
        final_status, code = run_pipeline(args=args, file_name=final_pipeline_file_name, code=code,
                                          schema_data=schema_data,
                                          run_mode=__execute_mode, sub_task=sub_task, iteration=iteration,
                                          time_total=time_total,
                                          time_catalog=time_catalog, time_generate=time_generate,
                                          all_token_count=all_token_count,
                                          prompt_token_count=prompt_token_count)

    return final_status, code


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
            save_text_file(fname=pipeline_fname, data=code.replace(args.data_source_train_path,"train.csv").replace(args.data_source_test_path, "test.csv"))
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
                                                                pipeline_error_class=result.get_error_class(),
                                                                pipeline_error_detail=result.get_error_detail(),
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
             time_catalog=time_catalog, time_generate=time_generate, time_total=time_total + time_execute,
             time_execute=time_execute,
             prompt_token_count=prompt_token_count, all_token_count=all_token_count, operation_tag='Run-Pipeline',
             run_mode=run_mode, results_verified=results_verified, results=results, final_status=final_status)

    return final_status, code


def compare_orig_and_clean_updates(orig_fname: str, clean_fname: str, cols_list_dtype):
    # check the data clean is available:
    if os.path.isfile(clean_fname):
        orig_data = pd.read_csv(orig_fname)
        clean_data = pd.read_csv(clean_fname)

        cols = orig_data.columns.to_list()
        total_refined_cols = 0
        refine_cols = []
        total_diffs = 0
        for c in cols:
            on = orig_data[c].nunique()
            if c not in cols_list_dtype.keys():
                cn = clean_data[c].nunique()
            else:
                cn = cols_list_dtype[c]
            if on - cn != 0:
                total_refined_cols += 1
                refine_cols.append(f"{c}#{on}#{cn}#{on - cn}")
                total_diffs += on - cn

        return {"total_refined_cols": total_refined_cols, "refine_cols": ";".join(refine_cols),
                "total_diffs": total_diffs}

    return None


def get_column_dummy_patches(column_name: str, dataframe_name: str):
    patch = [f"{dataframe_name}['{column_name}'] = {dataframe_name}['{column_name}'].str.split(',')"]
    patch.append(f"{dataframe_name} = {dataframe_name}.explode('{column_name}')")
    patch.append \
        (f"{dataframe_name} = pd.concat([{dataframe_name},pd.get_dummies({dataframe_name}['{column_name}'].str.strip(), prefix='{column_name}_')], axis=1)")
    patch.append(f"{dataframe_name} = {dataframe_name}.drop(columns=['{column_name}'])")
    return patch


def clean_categorical_data(args, data_profile_path: str, time_catalog: float = 0,
                           iteration: int = 1):
    from util.Config import __execute_mode, _llm_platform
    all_token_count = 0
    time_generate = 0
    time_total = 0

    cat = load_data_source_profile(data_source_path=data_profile_path,
                                   file_format="JSON",
                                   target_attribute=args.target_attribute,
                                   enable_reduction=args.enable_reduction,
                                   categorical_values_restricted_size=-1)
    data_cleaning_prompt = prompt_factory_data_cleaning(catalog=[cat])
    parts = data_cleaning_prompt.get_parts()
    destination_ds_name = args.data_source_path
    clean_fname = f"{destination_ds_name.replace('.csv', '')}_{_llm_platform}_clean.csv"

    final_status = False
    for pid in range(0, len(parts)):
        prompt_format = data_cleaning_prompt.format(part_id=pid)
        prompt_system_message = prompt_format["system_message"]
        prompt_user_message = prompt_format["user_message"]

        # Save prompt:
        prompt_file_name = f"{args.llm_model}-{data_cleaning_prompt.class_name}-iteration-{iteration}-part-{pid}"
        file_name = f'{args.output_path}/{prompt_file_name}'
        prompt_fname = f"{file_name}.prompt"
        save_prompt(fname=prompt_fname, system_message=prompt_system_message, user_message=prompt_user_message)

        # Generate LLM code
        code, prompt_token_count, time_tmp_gen = GenerateLLMCode.generate_llm_code(user_message=prompt_user_message,
                                                                                   system_message=prompt_system_message)
        time_generate_extra = 0
        for i in range(5):
            if code == "Insufficient information.":
                code, tokens_count, time_tmp_gen = GenerateLLMCode.generate_llm_code(user_message=prompt_user_message,
                                                                                     system_message=prompt_system_message)
                all_token_count += tokens_count
                time_generate_extra += time_tmp_gen
            else:
                break
        time_generate += time_tmp_gen
        cols_list_dtype = dict()
        if pid == len(parts) - 1:
            try:
                # fine dataframe name:
                df_name = None
                for line in code.splitlines():
                    if not line.startswith('#') and line.find(".to_csv"):
                        df_name = line.split(".to_csv")
                        df_name = df_name[0]
                cols_list_dtype = prompt_format["columns_list_dtype"]
                dummy_patch = []
                for cld in cols_list_dtype.keys():
                    dummy_patch.extend(get_column_dummy_patches(column_name=cld, dataframe_name=df_name))
                dummy_patch = "\n".join(dummy_patch)

                code = code.replace(f"{df_name}.to_csv", f"\n{dummy_patch}\n{df_name}.to_csv")
            except:
                pass

        pipeline_fname = f"{file_name}_draft.py"
        save_text_file(fname=pipeline_fname, data=code)

        final_status, _, destination_ds_name = run_data_cleaning_pipeline(args=args,
                                                                          file_name=file_name,
                                                                          orig_code=code,
                                                                          run_mode=__execute_mode,
                                                                          iteration=iteration,
                                                                          time_total=time_total,
                                                                          time_catalog=time_catalog,
                                                                          time_generate=time_generate,
                                                                          all_token_count=all_token_count,
                                                                          prompt_token_count=prompt_token_count,
                                                                          destination_ds_name=destination_ds_name,
                                                                          clean_fname=clean_fname,
                                                                          orig_fname=args.data_source_path,
                                                                          sub_ds_name="all-data",
                                                                          cols_list_dtype=cols_list_dtype)

    if final_status:
        split_clean_data_save(data_path=clean_fname, ds_name=args.dataset_name, out_path=args.root_data_path)


def run_data_cleaning_pipeline(args, file_name, orig_code, run_mode, iteration: int = 1,
                               time_total: int = 0, time_catalog: float = 0, time_generate: int = 0,
                               all_token_count: int = 0, prompt_token_count: int = 0, orig_fname: str = None,
                               destination_ds_name: str = None, clean_fname: str = None, sub_ds_name: str = None,
                               cols_list_dtype = None):
    time_execute = 0
    final_status = False

    # Run pipeline with original data
    code = orig_code.replace("original_data.csv", destination_ds_name)
    code = code.replace("clean_data.csv", clean_fname)

    iteration_error = 0
    refine_cols = None
    total_refined_cols = 0
    total_diffs = 0
    for i in range(0, args.prompt_number_iteration_error):
        time_start_1 = time.time()
        result = RunCode.execute_code(src=code, parse=None, run_mode=run_mode)
        time_end_1 = time.time()
        if result.get_status():
            time_execute = time_end_1 - time_start_1
            pipeline_fname = f"{file_name}-RUN.py"
            save_text_file(fname=pipeline_fname, data=code)
            final_status = True
            R = compare_orig_and_clean_updates(orig_fname=orig_fname, clean_fname=clean_fname, cols_list_dtype=cols_list_dtype)
            if R is not None:
                refine_cols = R['refine_cols']
                total_refined_cols = R['total_refined_cols']
                total_diffs = R['total_diffs']
            break
        else:
            error_fname = f"{file_name}_{i}_RUN.error"
            pipeline_fname = f"{file_name}_{i}_RUN.python"
            save_text_file(error_fname, f"{result.get_exception()}")
            save_text_file(fname=pipeline_fname, data=code)

            system_message, user_message = error_prompt_factory(pipeline_code=code,
                                                                pipeline_error_class=result.get_error_class(),
                                                                pipeline_error_detail=result.get_error_detail(),
                                                                schema_data='',
                                                                task_type='',
                                                                data_source_train_path='',
                                                                data_source_test_path='',
                                                                pipeline_type="data-cleaning"
                                                                )
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
    save_cleaning_log(args=args, iteration=iteration, iteration_error=iteration_error,
                      time_catalog=time_catalog, time_generate=time_generate, time_total=time_total + time_execute,
                      time_execute=time_execute, prompt_token_count=prompt_token_count, all_token_count=all_token_count,
                      operation_tag='Run-Data-Cleaning-Pipeline', final_status=final_status,
                      total_refined_cols=total_refined_cols, refine_cols=refine_cols, sub_dataset_name=sub_ds_name,
                      total_diffs=total_diffs)

    return final_status, code, clean_fname


def clean_data_catalog(args, data_profile_path: str, time_catalog: float = 0, iteration: int = 1):
    all_token_count = 0
    time_generate = 0
    time_total = 0

    cat = load_data_source_profile(data_source_path=data_profile_path,
                                   file_format="JSON",
                                   target_attribute=args.target_attribute,
                                   enable_reduction=args.enable_reduction,
                                   cleaning=True,
                                   categorical_values_restricted_size=-1)
    catalog_cleaning_prompt = prompt_factory_data_catalog_cleaning(catalog=[cat])
    prompt_format = catalog_cleaning_prompt.format()
    prompt_system_message = prompt_format["system_message"]
    prompt_user_message = prompt_format["user_message"]

    if prompt_user_message is None:
        return

    # Save prompt:
    prompt_file_name = f"{args.llm_model}-{catalog_cleaning_prompt.class_name}-iteration-{iteration}"
    file_name = f'{args.output_path}/{prompt_file_name}'
    prompt_fname = f"{file_name}.prompt"
    save_prompt(fname=prompt_fname, system_message=prompt_system_message, user_message=prompt_user_message)

    # Refine Catalog by LLM
    code, prompt_token_count, time_tmp_gen = DataPrepareLLM.data_prepare_llm(user_message=prompt_user_message,
                                                                             system_message=prompt_system_message)
    time_generate_extra = 0
    for i in range(5):
        if code == "Insufficient information.":
            code, tokens_count, time_tmp_gen = DataPrepareLLM.data_prepare_llm(user_message=prompt_user_message,
                                                                               system_message=prompt_system_message)
            all_token_count += tokens_count
            time_generate_extra += time_tmp_gen
        else:
            break
    time_generate += time_tmp_gen
    result = DataPrepareLLM.extract_catalog_values(result=code)
    if len(result) > 0:
        data = pd.read_csv(args.data_source_path)
        col_names = data.columns
        for k in result.keys():
            if result[k] in {'categorical', 'list'}:
                if k not in col_names:
                    k = k[1:len(k) - 1]
                column_values = data[k].dropna().unique().tolist()
                if result[k] == 'list':
                    column_values_list = []
                    for cv in column_values:
                        for ev in cv.split(','):
                            evv = ev.strip()
                            if len(evv) > 0:
                                column_values_list.append(evv)
                    column_values = set(column_values_list)
                piu = ProfileInfoUpdate(column_name=k, column_values=column_values, column_type=result[k])
                piu.save_profile(f"{data_profile_path}_update")
