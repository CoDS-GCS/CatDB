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

# Build TabPFN Classifier 
def runTabPFNClassifier(args, df_train, df_test, train_y, test_y):
  #try:
    print("-----------------------------")
    print(df_train[args.target_attribute])
    clf_no_feat_eng = TabPFNClassifier(device=('cuda' if torch.cuda.is_available() else 'cpu'), N_ensemble_configurations=4)
    clf_no_feat_eng.fit = partial(clf_no_feat_eng.fit, overwrite_warning=True)

    caafe_clf = CAAFEClassifier(base_classifier=clf_no_feat_eng,
                                llm_model=args.llm_model,
                                iterations=args.number_iteration)

    caafe_clf.fit_pandas(df_train,
                        target_column_name=args.target_attribute,
                        dataset_description=args.description)
    

    # pred_test = caafe_clf.predict(df_test)
    # pred_train = caafe_clf.predict(df_train)

    # acc_test = accuracy_score(pred_test, test_y)
    # acc_train = accuracy_score(pred_train, train_y)

    # train_F1_score = f1_score(train_y, pred_train)
    # test_F1_score = f1_score(test_y, pred_test)
    # status = True
  
  # except Exception as err:
  #   acc_train = -1
  #   train_F1_score = -1
  #   acc_test = -1
  #   test_F1_score = -1
  #   status = False  

  #  return status, acc_train, train_F1_score, acc_test, test_F1_score
  
    return True, 0,0,0,0


# Build RandomForest Classifier 
def runRandomForestClassifier(args, df_train, df_test, train_y, test_y):
  # try: 
    clf = RandomForestClassifier()
    caafe_clf = CAAFEClassifier(base_classifier=clf,
                                llm_model=args.llm_model,
                                iterations=args.number_iteration)

    caafe_clf.fit_pandas(df_train,
                        target_column_name=args.target_attribute,
                        dataset_description=args.description)
    

    # pred_test = caafe_clf.predict(df_test)
    # pred_train = caafe_clf.predict(df_train)

    # acc_test = accuracy_score(pred_test, test_y)
    # acc_train = accuracy_score(pred_train, train_y)

    # train_F1_score = f1_score(train_y, pred_train)
    # test_F1_score = f1_score(test_y, pred_test)
    # status = True

  # except Exception as err:
  #   acc_train = -1
  #   train_F1_score = -1
  #   acc_test = -1
  #   test_F1_score = -1
  #   status = False  
  #   print(err)
   
    #return status, acc_train, train_F1_score, acc_test, test_F1_score
    return True, 0,0,0,0

# Run CAAFE
def run_caafe(args):
  openai.api_key = os.environ.get("OPENAI_API_KEY")

  df_train = pd.read_csv(args.data_source_train_path)
  df_test = pd.read_csv(args.data_source_test_path)

  df_train, df_test = make_datasets_numeric(df_train=df_train, df_test=df_test, target_column=args.target_attribute)
  
  # print(df_train)
  X_train = df_train #.drop(columns=[args.target_attribute])
  y_train = df_train[args.target_attribute]

  X_test = df_test #.drop(columns=[args.target_attribute])
  y_test = df_test[args.target_attribute]

  # print(X_train)
  # print(y_train)

  #print(X_train)
  # X_train, y_train = data.get_X_y(df_train, args.target_attribute)
  # X_test, y_test = data.get_X_y(df_test, args.target_attribute) 

  print(X_train)
  print(y_train)

  if args.classifier == "TabPFN":
    status, acc_train, train_F1_score, acc_test, test_F1_score = runTabPFNClassifier(args, X_train, X_test, y_train, y_test)
  
  elif args.classifier == "RandomForest":
    status, acc_train, train_F1_score, acc_test, test_F1_score = runRandomForestClassifier(args, X_train, X_test, y_train, y_test)  

  # return status, acc_train, train_F1_score, acc_test, test_F1_score   

  return False, 0, 0, 0, 0 

def save_config(dataset_name,target, task_type, data_out_path):
    config_strs = [f"- name: {dataset_name}",
                       "  dataset:",
                       f"    train: \'{{user}}/data/{dataset_name}/{dataset_name}_train.csv\'",
                       f"    test: \'{{user}}/data/{dataset_name}/{dataset_name}_test.csv\'",
                       f"    target: {target}",
                       f"    type: {task_type}",
                       "  folds: 1",
                       "\n"]
    config_str = "\n".join(config_strs)

    yaml_file_local = f'{data_out_path}/{dataset_name}/{dataset_name}.yaml'
    f_local = open(yaml_file_local, 'w')
    f_local.write("--- \n \n")
    f_local.write(config_str)
    f_local.close()

if __name__ == '__main__':
   #args = parse_arguments()
   cc_test_datasets_multiclass = data.load_all_data()
  #  ds = cc_test_datasets_multiclass[0]
  #  print(len(ds[1]))

   data_path = "/home/saeed/Documents/Github/CatDB/Experiments/data/"

   myds = {"balance-scale": "Balance-Scale", "breast-w": "Breast-w", "cmc":"CMC",
           "credit-g":"Credit-g", "diabetes":"diabetes", "tic-tac-toe":"Tic-Tac-Toe", "eucalyptus":"Eucalyptus",
           "pc1":"PC1", "airlines":"Airlines", "jungle_chess_2pcs_raw_endgame_complete": "Jungle-Chess"}
   for d in cc_test_datasets_multiclass:   
      ds_name = d[0]

      ds_name = myds[ds_name]

      ds_path = f"{data_path}/{ds_name}"
      print(ds_path)
      os.makedirs(ds_path, exist_ok=True)
      ds, df_train, df_test, _, _ = data.get_data_split(d, seed=0)

      target_col = target_column_name = ds[4][-1]

      n_classes = df_train[target_col].nunique()
      if n_classes == 2:
          task_type = "binary"
      elif n_classes < 300 :
          task_type = "multiclass"
      else:
          task_type = "regression"       

      save_config(dataset_name=ds_name, target=target_col, data_out_path=data_path, task_type=task_type)
      df_train.to_csv(f'{ds_path}/{ds_name}_train.csv', index=False)
      df_test.to_csv(f'{ds_path}/{ds_name}_test.csv', index=False)


      desc_file = f'{ds_path}/{ds_name}.txt'
      f_local = open(desc_file, 'w')
      f_local.write(ds[-1])
      f_local.close()

      df_all = pd.concat([df_train, df_test])
      df_all.to_csv(f'{ds_path}/{ds_name}.csv', index=False)

      print(f"$CMD {ds_name} {task_type} test")


  #  start = time.time()
  #  status, acc_train, train_F1_score, acc_test, test_F1_score = run_caafe(args)
  #  end = time.time()

  #  execute_time = end - start

  #  if args.task_type != 'regression':
  #   try:
  #       df_result = pd.read_csv(args.log_file_name)
  #   except Exception as err:  
  #       df_result = pd.DataFrame(columns=["dataset_name",
  #                                         "config",
  #                                         "llm_model",
  #                                         "has_description",
  #                                         "classifier",
  #                                         "task_type",
  #                                         "status",
  #                                         "number_iteration",
  #                                         "pipeline_gen_time",
  #                                         "execution_time",
  #                                         "train_accuracy",
  #                                         "train_f1_score",
  #                                         "train_log_loss",
  #                                         "train_r_squared",
  #                                         "train_rmse",
  #                                         "test_accuracy",
  #                                         "test_f1_score",
  #                                         "test_log_loss",
  #                                         "test_r_squared",
  #                                         "test_rmse"])
    
  #   log = [args.dataset_name, 
  #          "CAAFE", 
  #          args.llm_model,
  #          args.dataset_description, 
  #          args.classifier, 
  #          args.task_type, 
  #          status, 
  #          args.number_iteration,
  #          -1, 
  #          execute_time, 
  #          acc_train, 
  #          train_F1_score, 
  #          -1, -1, -1, 
  #          acc_test, 
  #          test_F1_score, 
  #          -1, -1, -1]   

  #   df_result.loc[len(df_result)] = log
  #   df_result.to_csv(args.log_file_name, index=False)   