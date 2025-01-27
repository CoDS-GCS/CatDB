from argparse import ArgumentParser
from catalog.Catalog import load_data_source_profile_as_chunck
from catalog.Dependency import load_dependency_info
from pipegen.GeneratePipeLine import generate_and_verify_pipeline
from runcode.RunLocalPipeLine import run_local_pipeline_code
from util.Config import load_config
from util.FileHandler import read_text_file_line_by_line
import time
import yaml

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])
# install("category_encoders")

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
                args.data_source_verify_path = f"{args.root_data_path}/{config_data[0].get('dataset').get('train')}"
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


def load_privious_codes(chain_length: int, llm_model: str, class_name: str, path: str, extention: str):
    codes = [None for i in range(0, chain_length)]
    for i in range(0, chain_length):
        prompt_file_name = f"{llm_model}-{class_name}-No-iteration-{i+1}"
        file_name = f'{path}/{prompt_file_name}{extention}'
        try:
            codes[i] = read_text_file_line_by_line(fname=file_name)
        except Exception as err:
            pass
    return codes


def merge_chains(codes, privious_task_code, task, base_index, run_mode):
    merged_codes = [None for i in range(0, len(codes))]
    for index, pc in enumerate(codes):
        if pc is None:
            final_status = False
            repeat = 0
            if base_index == -1:
                iteration=index+1
            else:
                iteration = base_index
            while final_status == False and repeat < 10:
                final_status, code = generate_and_verify_pipeline(args=args, catalog=[catalogs[index]], run_mode=run_mode,
                                                                  sub_task=task,  time_catalog=time_catalog,
                                                                  iteration=iteration, dependency=dependencies, previous_result=privious_task_code)
                if final_status:
                    merged_codes[index] = code
                    privious_task_code = code
                repeat += 1
        else:
            merged_codes[index] = codes[index]
    return merged_codes

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
                api_config_path="APIKeys.yaml")
    catalogs = load_data_source_profile_as_chunck(data_source_path=data_profile_path,
                                                file_format="JSON",
                                                target_attribute=args.target_attribute,
                                                enable_reduction=args.enable_reduction,
                                                categorical_values_restricted_size=-1, chunk_size=25)
    time_end = time.time()
    time_catalog = time_end - time_start

    # Create Preprocessing Chain:
    class_name = f"{args.prompt_representation_type}{__sub_task_data_preprocessing}-{args.prompt_samples_type}-{args.prompt_number_samples}-SHOT"
    preprocessing_codes = load_privious_codes(chain_length=len(catalogs),llm_model=args.llm_model, class_name=class_name,path=args.output_path, extention='.py')
    preprocessing_codes = merge_chains(codes=preprocessing_codes, privious_task_code=None, task=__sub_task_data_preprocessing, base_index=-1, run_mode=__gen_verify_mode)

    # Create Feature Engineering Chain:
    class_name = f"{args.prompt_representation_type}{__sub_task_feature_engineering}-{args.prompt_samples_type}-{args.prompt_number_samples}-SHOT"
    fe_codes = load_privious_codes(chain_length=len(catalogs),llm_model=args.llm_model, class_name=class_name,path=args.output_path, extention='.py')
    for i in range(0, len(catalogs)):
        if fe_codes[i] is not None:
            continue
        privious_task_code = preprocessing_codes[i]
        fe_codes_tmp = [None for k in range(0, i+1)]
        for j in range(0, i+1):
            fe_codes_tmp = merge_chains(codes=fe_codes_tmp, privious_task_code=privious_task_code, task=__sub_task_feature_engineering, base_index=i+1, run_mode=__gen_verify_mode)
        fe_codes[i] = fe_codes_tmp[i]

    # Create Model Selection Chain:
    class_name = f"{args.prompt_representation_type}{__sub_task_model_selection}-{args.prompt_samples_type}-{args.prompt_number_samples}-SHOT"
    ms_codes = load_privious_codes(chain_length=len(catalogs),llm_model=args.llm_model, class_name=class_name,path=args.output_path, extention='-RUN.py')
    for i in range(0, len(catalogs)):
        if ms_codes[i] is not None:
            run_local_pipeline_code(args=args, code=ms_codes[i], run_mode=__execute_mode)
        else:
            privious_task_code = fe_codes[i]
            ms_codes_tmp = [None for k in range(0, i + 1)]
            for j in range(0, i + 1):
                run_mode = __execute_mode
                # if j == i:
                #     run_mode = __execute_mode
                # else:
                #     run_mode = __gen_verify_mode
                ms_codes_tmp = merge_chains(codes=ms_codes_tmp, privious_task_code=privious_task_code,
                                                task=__sub_task_model_selection, base_index=i + 1, run_mode=run_mode)
            ms_codes[i] = ms_codes_tmp[i]
