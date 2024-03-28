from argparse import ArgumentParser

from caafe import CAAFEClassifier # Automated Feature Engineering for tabular datasets
from tabpfn import TabPFNClassifier # Fast Automated Machine Learning method for small tabular datasets

import os
import openai
import torch
from caafe import data
from sklearn.metrics import accuracy_score
from tabpfn.scripts import tabular_metrics
from functools import partial
from caafe.preprocessing import make_datasets_numeric
import yaml
import pandas as pd

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--log-file-name', type=str, default=None)
    parser.add_argument('--number-iteration', type=int, default=1)

    args = parser.parse_args()

     # read .yaml file and extract values:
    with open(args.metadata_path, "r") as f:
        try:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
            args.dataset_name = config_data[0].get('name')
            args.target_attribute = config_data[0].get('dataset').get('target')
            args.task_type = config_data[0].get('dataset').get('type')
            try:
                args.data_source_train_path = "../../../" + config_data[0].get('dataset').get('train').replace("{user}/", "")
                args.data_source_test_path = "../../../" + config_data[0].get('dataset').get('test').replace("{user}/","")
            except Exception as ex:
                raise Exception(ex)

            try:
                args.number_folds = int(config_data[0].get('folds'))
            except yaml.YAMLError as ex:
                args.number_folds = 1

        except yaml.YAMLError as ex:
            raise Exception(ex)
    
    if args.log_file_name is None:
        raise Exception("--log-file-name is a required parameter!")

    return args

def run_caafe(args):
  openai.api_key = os.environ.get("OPENAI_API_KEY")

  df_train = pd.read_csv(args.data_source_train_path)
  df_test = pd.read_csv(args.data_source_test_path)

  df_train, df_test = make_datasets_numeric(df_train, df_test, args.target_attribute)
  train_x, train_y = data.get_X_y(df_train, args.target_attribute)
  test_x, test_y = data.get_X_y(df_test, args.target_attribute)

  clf_no_feat_eng = TabPFNClassifier(device=('cuda' if torch.cuda.is_available() else 'cpu'), N_ensemble_configurations=4)
  clf_no_feat_eng.fit = partial(clf_no_feat_eng.fit, overwrite_warning=True)

  clf_no_feat_eng.fit(train_x, train_y)
  pred_test = clf_no_feat_eng.predict(test_x)
  pred_train = clf_no_feat_eng.predict(train_x)

  acc_test_before = accuracy_score(pred_test, test_y)
  acc_train_before = accuracy_score(pred_train, train_y)

  caafe_clf = CAAFEClassifier(base_classifier=clf_no_feat_eng,
                            llm_model="gpt-4",
                            iterations=args.number_iteration)

  caafe_clf.fit_pandas(df_train,
                      target_column_name=args.target_attribute,
                      dataset_description="There is no description for this dataset.")
  

  pred_test = caafe_clf.predict(df_test)
  pred_train = caafe_clf.predict(df_train)

  acc_test_after = accuracy_score(pred_test, test_y)
  acc_train_after = accuracy_score(pred_train, train_y)
  
  
  log = [args.dataset_name, args.task_type, acc_train_before, acc_test_before, acc_train_after, acc_test_after]
  
  return log    

if __name__ == '__main__':
   args = parse_arguments()
   log = run_caafe(args)

   try:
       df_result = pd.read_csv(args.log_file_name)
   except Exception as err:  
       df_result = pd.DataFrame(columns=["dataset_name","task_type", "accuracy_train_before", "accuracy_test_before", "accuracy_train_after", "accuracy_test_after"])

   df_result.loc[len(df_result)] = log
   df_result.to_csv(args.log_file_name, index=False)   