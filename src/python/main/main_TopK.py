from argparse import ArgumentParser
from catalog.Catalog import load_data_source_profile_TopK
from catalog.Dependency import load_dependency_info
from pipegen.GeneratePipeLine import generate_and_verify_pipeline
from runcode.RunLocalPipeLine import run_local_pipeline_code
from util.Config import load_config
from util.FileHandler import read_text_file_line_by_line
import time
import yaml

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
    parser.add_argument('--topk', type=int, default=10)
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
                args.data_source_train_clean_path = f"{args.data_source_train_path.replace('.csv','')}_clean.csv"
                args.data_source_test_clean_path = f"{args.data_source_test_path.replace('.csv','')}_clean.csv"
                args.data_source_verify_clean_path = f"{args.data_source_verify_path.replace('.csv','')}_clean.csv"
                args.data_source_clean_path = f"{args.data_source_path.replace('.csv', '')}_clean.csv"

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
    begin_iteration = 1
    end_iteration = 1

    data_profile_path = f"{args.catalog_path}/data_profile"
    time_start = time.time()
    dependency_file = f"{args.catalog_path}/dependency.yaml"
    dependencies = load_dependency_info(dependency_file=dependency_file, datasource_name=args.dataset_name)
    load_config(system_log=args.system_log, llm_model=args.llm_model, rules_path="Rules.yaml", evaluation_acc=False,
                api_config_path="APIKeys.yaml", enable_cache=False)

    k = args.topk
    flag = True
    while True:
        catalog = load_data_source_profile_TopK(data_source_path=data_profile_path,
                                                    file_format="JSON",
                                                    target_attribute=args.target_attribute,
                                                    enable_reduction=args.enable_reduction,
                                                    categorical_values_restricted_size=-1, k=k)
        time_end = time.time()
        time_catalog = time_end - time_start
        ti = 0
        t = args.prompt_number_iteration * 2
        while begin_iteration < args.prompt_number_iteration + end_iteration:
            final_status, code = generate_and_verify_pipeline(args=args, catalog=[catalog], run_mode=__execute_mode,
                                time_catalog=time_catalog, iteration=k, dependency=dependencies)
            if final_status:
                begin_iteration += 1

            ti += 1
            if ti > t:
                break
        k += args.topk
        if k >= catalog.ncols and flag:
            k = catalog.ncols
            flag = False
        elif flag == False:
            break