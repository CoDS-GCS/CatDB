from argparse import ArgumentParser

from caafe import CAAFEClassifier # Automated Feature Engineering for tabular datasets
from tabpfn import TabPFNClassifier # Fast Automated Machine Learning method for small tabular datasets
from sklearn.ensemble import RandomForestClassifier

import os
import openai
import torch
from caafe import data
from sklearn.metrics import accuracy_score
from tabpfn.scripts import tabular_metrics
from functools import partial
from caafe.preprocessing import make_datasets_numeric

import pandas as pd

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default=None)
    parser.add_argument('--log-file-name', type=str, default=None)

    args = parser.parse_args()

    if args.dataset_name is None:
        raise Exception("--dataset-name is a required parameter!")
    
    if args.log_file_name is None:
        raise Exception("--log-file-name is a required parameter!")

    return args

def run_caafe(dataset_name):
  openai.api_key = os.environ.get("OPENAI_API_KEY")

  metric_used = tabular_metrics.auc_metric
  cc_test_datasets_multiclass = data.load_all_data()

  ds = None
  for d in cc_test_datasets_multiclass:
     if dataset_name in d[0]:
        ds = d
        break
  if ds is None:
     return None      
  #ds = cc_test_datasets_multiclass[dataset_id]
  ds_name = ds[0]
  ds, df_train, df_test, _, _ = data.get_data_split(ds, seed=0)
  target_column_name = ds[4][-1]
  dataset_description = ds[-1]

  target_col = target_column_name = ds[4][-1]
  n_classes = df_train[target_col].nunique()
  if n_classes == 2:
      task_type = "binary"
  elif n_classes < 300 :
    task_type = "multiclass"
  else:
    task_type = "regression"       

  df_train, df_test = make_datasets_numeric(df_train, df_test, target_column_name)
  train_x, train_y = data.get_X_y(df_train, target_column_name)
  test_x, test_y = data.get_X_y(df_test, target_column_name)

  clf_no_feat_eng = TabPFNClassifier(device=('cuda' if torch.cuda.is_available() else 'cpu'), N_ensemble_configurations=4)
  clf_no_feat_eng.fit = partial(clf_no_feat_eng.fit, overwrite_warning=True)

  clf_no_feat_eng.fit(train_x, train_y)
  pred = clf_no_feat_eng.predict(test_x)
  acc_before = accuracy_score(pred, test_y)

  # print(f'Accuracy before CAAFE {acc}')

  # caafe_clf = CAAFEClassifier(base_classifier=clf_no_feat_eng,
  #                           llm_model="gpt-4",
  #                           iterations=10)

  # caafe_clf.fit_pandas(df_train,
  #                     target_column_name=target_column_name,
  #                     dataset_description=dataset_description)

  # pred = caafe_clf.predict(df_test)
  # acc_after = accuracy_score(pred, test_y)
  # # print(f'Accuracy after CAAFE {acc}')

  acc_after = 0

  log = [ds_name,task_type,acc_before,acc_after]
  
  return log    

if __name__ == '__main__':
   args = parse_arguments()
   log = run_caafe(dataset_name=args.dataset_name)

   if log is not None:
    df_result = pd.DataFrame(columns=["dataset_name","task_type", "accuracy_before", "accuracy_after"])
    df_result.loc[len(df_result)] = log
    df_result.to_csv(args.log_file_name, index=False)   