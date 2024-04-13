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

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--results-path', type=str, default=None)
    parser.add_argument('--out-path', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.results_path is None:
        raise Exception("--results-path is a required parameter!") 
    
    if args.out_path is None:
        raise Exception("--out-path is a required parameter!")     
    
    return args



if __name__ == '__main__':
   args = parse_arguments()
   log = run_caafe(args)

   try:
       df_result = pd.read_csv(args.log_file_name)
   except Exception as err:  
       df_result = pd.DataFrame(columns=["dataset_name","task_type", "llm_model","has_description","prompt_representation_type","prompt_example_type","prompt_number_example","number_tokens","number_bool","number_int","number_float","number_string"])

   df_result.loc[len(df_result)] = log
   df_result.to_csv(args.log_file_name, index=False)   