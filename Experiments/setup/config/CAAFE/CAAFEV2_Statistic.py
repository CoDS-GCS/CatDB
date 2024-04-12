from argparse import ArgumentParser

from caafe import CAAFEClassifier # Automated Feature Engineering for tabular datasets
import caafe.caafe
from tabpfn import TabPFNClassifier # Fast Automated Machine Learning method for small tabular datasets

import os
import openai
import torch
from caafe import data
from tabpfn.scripts import tabular_metrics
from sklearn.metrics import accuracy_score
from functools import partial
from caafe.preprocessing import make_datasets_numeric
import yaml
import pandas as pd
import tiktoken
import caafe
import numpy as np
import copy
from sklearn.model_selection import train_test_split

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
    
    if args.dataset_description.lower() == "yes":
        dataset_description_path = args.metadata_path.replace(".yaml", ".txt")
        args.description = read_text_file_line_by_line(fname=dataset_description_path)
        args.dataset_description = 'yes'
    else:
        args.description = "There is not data description for this dataset."
        args.dataset_description = 'no'

    return args

def get_number_tokens(prompt, args):
    enc = tiktoken.get_encoding("cl100k_base")
    enc = tiktoken.encoding_for_model(args.llm_model)
    token_integers = enc.encode(prompt)
    num_tokens = len(token_integers)

    return num_tokens

def run_caafe(args):
  
  df_train_old = pd.read_csv(args.data_source_train_path)
  df_test_old = pd.read_csv(args.data_source_test_path)

  df_train, df_test = make_datasets_numeric(df_train_old, df_test_old, args.target_attribute)
  feature_columns = list(df_train_old.drop(columns=[args.target_attribute]).columns)
  feature_names = list(feature_columns)

  X, y = (
            df_train_old.drop(columns=[args.target_attribute]).values,
            df_train_old[args.target_attribute].values,
        )
  
  ds = ["dataset",
        X,
        y,
        [],
        feature_names + [args.target_attribute],
        {},
        args.description,
        ]
  caafe_promt = caafe.caafe.build_prompt_from_df(df=df_train, ds=ds, iterative=10)  
  
  number_tokens = get_number_tokens(prompt=caafe_promt, args=args)
  log = [args.dataset_name, 
         args.task_type, 
         args.llm_model,
         args.dataset_description, 
         "CAAFE",
         "Random",
         10,
         number_tokens,
         0,0,0,0]
  
  return log 



if __name__ == '__main__':
   args = parse_arguments()
   log = run_caafe(args)

   try:
       df_result = pd.read_csv(args.log_file_name)
   except Exception as err:  
       df_result = pd.DataFrame(columns=["dataset_name","task_type", "llm_model","has_description","prompt_representation_type","prompt_example_type","prompt_number_example","number_tokens","number_bool","number_int","number_float","number_string"])

   df_result.loc[len(df_result)] = log
   df_result.to_csv(args.log_file_name, index=False)   