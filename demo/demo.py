
from argparse import ArgumentParser
import yaml

def get_args(dataset_name):
  parser = ArgumentParser()
  parser.add_argument('-f')
  
  args = parser.parse_args() 
  args.dataset_name = dataset_name
  args.metadata_path = f"/content/demo/data/{args.dataset_name}/{args.dataset_name}.yaml"
  args.root_data_path = f"/content/demo/data/"
  args.catalog_path = f"/content/demo/catalog/{args.dataset_name}"
  args.prompt_number_iteration = 1
  args.prompt_number_iteration_error = 10
  args.output_path = "/content/demo/catdb-results/"
  args.llm_model = "gemini-1.5-pro-latest"
  args.result_output_path = "/content/demo/catdb-results/results.csv"
  args.error_output_path = "/content/demo/catdb-results/error.csv"
  args.system_log = "/content/demo/catdb-results/system-log.dat"
  args.enable_reduction = True
  args.APIKeys_File='/content/catdb-setup/APIKeys.yaml'
  args.data_profile_path = f"{args.catalog_path}/data_profile"
  args.prompt_representation_type = 'CatDB'
  args.prompt_samples_type = 'Random'
  args.prompt_number_samples = 0
  args.description = ''
  args.dataset_description = 'No'
  
  with open(args.metadata_path, "r") as f:
    try:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
        args.target_attribute = config_data[0].get('dataset').get('target')
        args.task_type = config_data[0].get('dataset').get('type')
        args.multi_table = config_data[0].get('dataset').get('multi_table')
        args.target_table = config_data[0].get('dataset').get('target_table')
        try:
          args.data_source_train_path = f"{args.root_data_path}/{config_data[0].get('dataset').get('train')}"
          args.data_source_test_path = f"{args.root_data_path}/{config_data[0].get('dataset').get('test')}"
          args.data_source_verify_path = f"{args.root_data_path}/{config_data[0].get('dataset').get('verify')}"
        except Exception as ex:
            raise Exception(ex)

    except yaml.YAMLError as ex:
      raise Exception(ex)

  return args   