from argparse import ArgumentParser
from catalog.Catalog import load_data_source_profile
from catalog.Dependency import load_dependency_info
from pipegen.GeneratePipeLine import generate_and_verify_pipeline, run_pipeline, clean_categorical_data
from util.FileHandler import read_text_file_line_by_line
from util.Config import load_config
from pipegen.Metadata import Metadata
import time
import yaml
import os


def parse_arguments():
    from util.Config import CATEGORICAL_VALUES_RESTRICTED_SIZE
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
    parser.add_argument('--categorical-values-restricted-size', type=int, default=CATEGORICAL_VALUES_RESTRICTED_SIZE)

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
                args.data_source_path = f"{args.root_data_path}/{args.dataset_name}/{args.dataset_name}.csv"
                args.data_source_train_path = f"{args.root_data_path}/{config_data[0].get('dataset').get('train')}"
                args.data_source_test_path = f"{args.root_data_path}/{config_data[0].get('dataset').get('test')}"
                args.data_source_verify_path = f"{args.root_data_path}/{config_data[0].get('dataset').get('verify')}"
                args.data_source_train_clean_path = f"{args.data_source_train_path.replace('.csv','')}_LLM_clean.csv"
                args.data_source_test_clean_path = f"{args.data_source_test_path.replace('.csv','')}_LLM_clean.csv"
                args.data_source_verify_clean_path = f"{args.data_source_verify_path.replace('.csv','')}_LLM_clean.csv"
                args.data_source_clean_path = f"{args.data_source_path.replace('.csv', '')}_LLM_clean.csv"

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
                                           dependency=dependencies[tbl],
                                           categorical_values_restricted_size=-1)
            cat.table_name = tbl
            catalog.append(cat)
    else:
        load_config(system_log=args.system_log, llm_model=args.llm_model, rules_path="Rules.yaml")
        # check the data clean is available:
        from util.Config import _llm_platform
        if os.path.isfile(args.data_source_train_clean_path):
            args.data_source_train_path = args.data_source_train_clean_path.replace("LLM", _llm_platform)

        if os.path.isfile(args.data_source_test_clean_path):
            args.data_source_test_path = args.data_source_test_clean_path.replace("LLM", _llm_platform)

        if os.path.isfile(args.data_source_verify_clean_path):
            args.data_source_verify_path = args.data_source_verify_clean_path.replace("LLM", _llm_platform)

        catalog.append(load_data_source_profile(data_source_path=data_profile_path,
                                                file_format="JSON",
                                                target_attribute=args.target_attribute,
                                                enable_reduction=args.enable_reduction,
                                                categorical_values_restricted_size=-1))

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
