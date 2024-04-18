from argparse import ArgumentParser

from caafe import CAAFEClassifier # Automated Feature Engineering for tabular datasets
from tabpfn import TabPFNClassifier # Fast Automated Machine Learning method for small tabular datasets

import os
import openai
import torch
from caafe import data
from sklearn.metrics import accuracy_score, f1_score
from tabpfn.scripts import tabular_metrics
from functools import partial
from caafe.preprocessing import make_datasets_numeric
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import accuracy_score, log_loss

def read_text_file_line_by_line(fname:str):
    try:
        with open(fname) as f:
            lines = f.readlines()
            raw = "".join(lines)
            return raw
    except Exception as ex:
        raise Exception (f"Error in reading file:\n {ex}")

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--log-file-name', type=str, default=None)
    parser.add_argument('--dataset-description', type=str, default="yes")
    parser.add_argument('--number-iteration', type=int, default=1)
    parser.add_argument('--llm-model', type=str, default=None)
    parser.add_argument('--classifier', type=str, default=None)

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
    
    if args.llm_model is None:
        raise Exception("--llm-model is a required parameter!") 
    
    if args.classifier is None:
        raise Exception("--classifier is a required parameter!") 
    
    if args.dataset_description.lower() == "yes":
        dataset_description_path = args.metadata_path.replace(".yaml", ".txt")
        args.description = read_text_file_line_by_line(fname=dataset_description_path)
        args.dataset_description = 'Yes'
    else:
        args.description = "There is not data description for this dataset."
        args.dataset_description = 'No'   

    return args

# Run CAAFE
def run_caafe(args):
      openai.api_key = os.environ.get("OPENAI_API_KEY")

      df_train = pd.read_csv(args.data_source_train_path).dropna()
      df_test = pd.read_csv(args.data_source_test_path).dropna()

      if args.classifier == "TabPFN":
          if len(df_train) > 10000:
            df_train = df_train.sample(frac=1).reset_index(drop=True)[0:10000]

          if len(df_test) > 10000:
            df_test = df_test.sample(frac=1).reset_index(drop=True)[0:10000]   

      df_train, df_test = make_datasets_numeric(df_train, df_test, args.target_attribute)
      _, train_y = data.get_X_y(df_train, args.target_attribute)
      _, test_y = data.get_X_y(df_test, args.target_attribute)

    #try:
      clf_no_feat_eng = None
      if args.classifier == "TabPFN":
        clf_no_feat_eng = TabPFNClassifier(device=('cuda' if torch.cuda.is_available() else 'cpu'), N_ensemble_configurations=4)    
      
      elif args.classifier == "RandomForest":
        clf_no_feat_eng = RandomForestClassifier(max_leaf_nodes=500)

      caafe_clf = CAAFEClassifier(base_classifier=clf_no_feat_eng,
                                    llm_model=args.llm_model,
                                    iterations=args.number_iteration)

      caafe_clf.fit_pandas(df_train,
                            target_column_name=args.target_attribute,
                            dataset_description=args.description)
        

      pred_test = caafe_clf.predict(df_test)
      pred_train = caafe_clf.predict(df_train)

      acc_test = accuracy_score(pred_test, test_y)
      acc_train = accuracy_score(pred_train, train_y)

      train_F1_score = -1
      test_F1_score = -1
      train_log_loss = -1
      test_log_loss = -1
      
      if args.task_type == "binary":
        train_F1_score = f1_score(train_y, pred_train)
        test_F1_score = f1_score(test_y, pred_test)
      
      elif args.task_type == "multiclass":
        y_train_prob = caafe_clf.predict_proba(df_train.drop(args.target_attribute, axis=1))
        y_test_prob = caafe_clf.predict_proba(df_test.drop(args.target_attribute, axis=1))
        train_log_loss = log_loss(train_y, y_train_prob)
        test_log_loss = log_loss(test_y, y_test_prob)

      status = True  
    # except Exception as err:
    #   pred_test = -1
    #   acc_train = -1
    #   train_F1_score = -1
    #   acc_test = -1
    #   test_F1_score = -1
    #   train_log_loss = -1
    #   test_log_loss = -1
    #   status = False

      return status, acc_train, train_F1_score, train_log_loss, acc_test, test_log_loss, test_F1_score

if __name__ == '__main__':
   args = parse_arguments()

   start = time.time()
   status, acc_train, train_F1_score, train_log_loss, acc_test, test_log_loss, test_F1_score = run_caafe(args)
   end = time.time()

   execute_time = end - start

   if args.task_type != 'regression':
    try:
        df_result = pd.read_csv(args.log_file_name)
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
    
    log = [args.dataset_name, 
           "CAAFE", 
           args.llm_model,
           args.dataset_description, 
           args.classifier, 
           args.task_type, 
           status, 
           args.number_iteration,
           -1, 
           execute_time, 
           acc_train, 
           train_F1_score, 
           train_log_loss, -1, -1, 
           acc_test, 
           test_F1_score, 
           test_log_loss, -1, -1]   

    df_result.loc[len(df_result)] = log
    df_result.to_csv(args.log_file_name, index=False)   