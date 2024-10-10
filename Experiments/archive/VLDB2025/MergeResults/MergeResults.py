import os
import sys
import pandas as pd
import numpy as np
import csv

dataset_corr = {"oml_dataset_12_rnc":"CatDB", 
                "oml_dataset_5_rnc":"CatDBChain",
                "oml_dataset_6_rnc":"CatDB",
                "oml_dataset_2_rnc":"CatDB",
                "oml_dataset_4_rnc":"CatDB",
                "oml_dataset_33_rnc":"CatDBChain",
                "oml_dataset_1_rnc":"CatDBChain",
                "oml_dataset_20_rnc":"CatDBChain",
                "oml_dataset_10_rnc":"CatDB",
                "oml_dataset_3_rnc":"CatDBChain",
                "oml_dataset_11_rnc":"CatDB",
                "oml_dataset_19_rnc":"CatDBChain",
                "oml_dataset_34_rnc":"CatDBChain",
                "oml_dataset_35_rnc":"CatDBChain",
                "oml_dataset_21_rnc":"CatDB",
                "oml_dataset_22_rnc":"CatDBChain",
                "oml_dataset_24_rnc":"CatDBChain",
                "oml_dataset_23_rnc":"CatDB",
                "Airline":"CatDBChain",
                "IMDB-IJS":"CatDBChain",
                "Accidents":"CatDBChain",
                "Financial":"CatDBChain",
                "Yelp":"CatDBChain",
                "EU-IT":"CatDB",
                "Etailing":"CatDB",
                "Midwest-Survey":"CatDB",
                "WiFi":"CatDB",
                "Utility":"CatDB"}
                

def load_results(path):
    df = pd.read_csv(path, low_memory=False, encoding='utf-8')
    if "number_of_samples" not in df.columns:
        df["number_of_samples"] = 0
    return df

def mergse_dfs(df_base, df_new):
    if len(df_new) > 0:
        for index, row in df_new.iterrows():
            df_base.loc[len(df_base)] = row 
    return df_base

def merge_raw_data(df_result, df_tmp):
    for index, row in df_tmp.iterrows():
        df_result.loc[len(df_result)] = row 
    return df_result    


def load_merge_all_results(root_path):
    columns = ["dataset_name", 
               "config", 
               "sub_task", 
               "llm_model", 
               "classifier", 
               "task_type", 
               "status",
               "number_iteration",
               "number_iteration_error", 
               "has_description", 
               "time_catalog_load", 
               "time_pipeline_generate",
               "time_total", 
               "time_execution", 
               "train_auc",
               "train_auc_ovo",
               "train_auc_ovr", 
               "train_accuracy",
               "train_f1_score", 
               "train_log_loss", 
               "train_r_squared", 
               "train_rmse", 
               "test_auc",
               "test_auc_ovo",
               "test_auc_ovr", 
               "test_accuracy", 
               "test_f1_score", 
               "test_log_loss", 
               "test_r_squared", 
               "test_rmse",
               "prompt_token_count",
               "all_token_count", 
               "operation",
               "number_of_samples"]

    results_path = [f"{root_path}/raw_results/Cleaning-Experiment1_LLM_CAAFE.dat",
                    f"{root_path}/raw_results/Cleaning-Experiment1_LLM_Pipe_Gen_CatDB.dat",
                    f"{root_path}/raw_results/Cleaning-Experiment1_LLM_Pipe_Gen_CatDBChain.dat",
                    #f"{root_path}/raw_results/",  # <<< CAAFE
                    f"{root_path}/raw_results/MicroBench-Experiment1_LLM_Pipe_Gen_CatDB.dat",
                    f"{root_path}/raw_results/MicroBench-Experiment1_LLM_Pipe_Gen_CatDBChain.dat",
                    f"{root_path}/raw_results/Multitable-Experiment1_LLM_CAAFE.dat",
                    f"{root_path}/raw_results/Multitable-Experiment1_LLM_Pipe_Gen_CatDB.dat",
                    f"{root_path}/raw_results/Multitable-Experiment1_LLM_Pipe_Gen_CatDBChain.dat",
                    f"{root_path}/raw_results/Experiment3_AutoML.dat"
                    ] 
    df_merge = pd.DataFrame(columns = columns)
    
    for rp in results_path:
        df_tmp = load_results(rp)    
        df_merge = merge_raw_data(df_result=df_merge, df_tmp=df_tmp)

    #df_merge = merge_raw_data(df_tmp=load_merge_AutoML_results(root_path=root_path), df_result=df_merge)
    return df_merge

def load_merge_AutoML_results(root_path):
    columns = ["dataset_name", "config", "sub_task", "llm_model", "classifier", "task_type", "status",
                "number_iteration","number_iteration_error", "has_description", "time_catalog_load", "time_pipeline_generate",
                "time_total", "time_execution", "train_auc","train_auc_ovo","train_auc_ovr", "train_accuracy",
                "train_f1_score", "train_log_loss", "train_r_squared", "train_rmse", "test_auc","test_auc_ovo",
                "test_auc_ovr", "test_accuracy", "test_f1_score", "test_log_loss", "test_r_squared", "test_rmse",
                "prompt_token_count","all_token_count", "operation","number_of_samples"]

    results_path_CatDB = [f"{root_path}/Experiment3_AutoML_CatDB.dat"] 
    
    results_path_CatDBChain = [f"{root_path}/Experiment3_AutoML_CatDBChain.dat"]    

    
    df_merge_CatDB = pd.DataFrame(columns = columns)
    df_merge_CatDBChain = pd.DataFrame(columns = columns)
    df_merge = pd.DataFrame(columns = columns)
    
    for rp in results_path_CatDB:
        df_tmp = load_results(rp)    
        df_merge_CatDB = merge_raw_data(df_result=df_merge_CatDB, df_tmp=df_tmp)

    for rp in results_path_CatDBChain:
        df_tmp = load_results(rp)    
        df_merge_CatDBChain = merge_raw_data(df_result=df_merge_CatDBChain, df_tmp=df_tmp)    

    for ds in dataset_corr.keys():
        data_corr = dataset_corr[ds]
        if data_corr == "CatDB":
            df_pip_gen = df_merge_CatDB
        else:
            df_pip_gen = df_merge_CatDBChain    

        df_tmp = df_pip_gen.loc[(df_pip_gen['dataset_name'] == ds)]
        df_merge = merge_raw_data(df_result=df_merge, df_tmp=df_tmp)

    return df_merge


def load_merde_AUTO_results(root_path, df):   

    results_path = [f"{root_path}/Experiment1_LLM_Pipe_Gen_AUTO.dat"]    
   
    
    for rp in results_path:
        df_tmp = load_results(rp)    
        df = merge_raw_data(df_result=df, df_tmp=df_tmp)

    return df

def load_merge_all_errors(root_path):
    columns = ["row_id", "dataset_name", "llm_model", "config", "sub_task", "error_class", "error_type",
                        "error_value", "error_detail", "error_exception", "file_name", "timestamp"]

    results_path = [#f"{root_path}/server-16/Error/LLM_Pipe_Error_1.dat",
                    f"{root_path}/server-16/Error/LLM_Pipe_Error_2.dat",
                    f"{root_path}/server-35/Error/LLM_Pipe_Error_1.dat",
                    f"{root_path}/server-35/Error/LLM_Pipe_Error_2.dat",
                    f"{root_path}/server-65/Error/LLM_Pipe_Error_1.dat",
                    f"{root_path}/server-65/Error/LLM_Pipe_Error_2.dat",
                    f"{root_path}/server-65/Error/LLM_Pipe_Error_3.dat",
                    f"{root_path}/server-113/Error/LLM_Pipe_Error_1.dat",
                    ] 
    #results_path = [f"{root_path}/server-16/Error/LLM_Pipe_Error_2.dat"] 
      
    df_merge = pd.DataFrame(columns = columns)
    
    for rp in results_path:
        df_tmp = load_results(rp)    
        df_merge = merge_raw_data(df_result=df_merge, df_tmp=df_tmp)

    return df_merge

def get_top_k_all(df, config, k):
    df_binary = df.loc[(df['train_auc'] >=0) &
                       (df['test_auc'] >=0) &
                       (df['task_type'] =='binary')]
    
    df_multi = df.loc[(df['train_auc_ovr'] >=0) &
                        (df['test_auc_ovr'] >=0) &
                        (df['task_type'] =='multiclass')]  
    df_multi['train_auc'] = df_multi['train_auc_ovr']
    df_multi['test_auc'] = df_multi['test_auc_ovr']
    
    df_multi_euit = df.loc[(df['train_accuracy'] >=0) &
                        (df['test_accuracy'] >=0) &
                        (df['dataset_name'] =='EU-IT')]  
    df_multi_euit['train_auc'] = df_multi_euit['train_accuracy']
    df_multi_euit['test_auc'] = df_multi_euit['test_accuracy']

    df_reg = df.loc[(df['train_r_squared'] >=0) & (df['train_r_squared'] <=1) &
                     (df['test_r_squared'] >=0) & (df['test_r_squared'] <=1) &
                     (df['task_type'] =='regression')]
    
    df_reg['train_auc'] = df_reg['train_r_squared']
    df_reg['test_auc'] = df_reg['test_r_squared']    

    df_merge = pd.DataFrame(columns = df_binary.columns)
    df_merge = merge_raw_data(df_result=df_merge, df_tmp=df_binary)
    df_merge = merge_raw_data(df_result=df_merge, df_tmp=df_multi)
    df_merge = merge_raw_data(df_result=df_merge, df_tmp=df_multi_euit)
    df_merge = merge_raw_data(df_result=df_merge, df_tmp=df_reg)

    if config not in {"CatDB", "CatDBChain"}:
        return df_merge
    else:
       df_merge = df_merge.sort_values(by='test_auc', ascending=False).reset_index(drop=True)
       return  df_merge.head(k)


def get_top_k_binary(df, config, k):
    df_binary = df.loc[(df['train_auc'] >=0) &
                       (df['test_auc'] >=0) &
                       (df['task_type'] =='binary')]
    if config not in {"CatDB", "CatDBChain"}:
        return df_binary
    else:
       df_binary = df_binary.sort_values(by='test_auc', ascending=False).reset_index(drop=True)
       return  df_binary.head(k)

def get_top_k_multiclass(df, config,k):
     df_multi = df.loc[(df['train_auc_ovr'] >=0) &
                        (df['test_auc_ovr'] >=0) &
                        (df['task_type'] =='multiclass')]  

     if config not in {"CatDB", "CatDBChain"}:
        return df_multi
     else:
       df_multi = df_multi.sort_values(by='test_auc_ovr', ascending=False).reset_index(drop=True)
       return  df_multi.head(k) 

def get_top_k_multiclass_EUIT(df, config,k):
     df_multi = df.loc[(df['train_accuracy'] >=0) &
                        (df['test_accuracy'] >=0) &
                        (df['task_type'] =='multiclass')]  
     df_multi['train_auc_ovr'] = df_multi['train_accuracy']
     df_multi['test_auc_ovr'] = df_multi['test_accuracy']

     if config not in {"CatDB", "CatDBChain"}:
        return df_multi
     else:
       df_multi = df_multi.sort_values(by='test_accuracy', ascending=False).reset_index(drop=True)
       return  df_multi.head(k)     

def get_top_k_regression(df, config,k):
     df_reg = df.loc[(df['train_r_squared'] >=0) & (df['train_r_squared'] <=1) &
                     (df['test_r_squared'] >=0) & (df['test_r_squared'] <=1) &
                     (df['task_type'] =='regression')]
     
     if config not in {"CatDB", "CatDBChain"}:
        return df_reg
     else:
       
       df_reg = df_reg.sort_values(by='test_r_squared', ascending=False).reset_index(drop=True)       
       return  df_reg.head(k) 
     
def get_top_k_chain(df, sub_task,k):
     df_task = df.loc[df['sub_task'] == sub_task]
     df_task = df_task.sort_values(by='number_iteration_error', ascending=True).reset_index(drop=True)       
     return df_task.head(k)      
     

def read_text_file_line_by_line(fname:str):
    try:
        with open(fname) as f:
            lines = f.readlines()
            raw = "".join(lines)
            return raw
    except Exception as ex:
        raise Exception (f"Error in reading file:\n {ex}")


def save_text_file(fname: str, data):
    try:
        f = open(fname, 'w')
        f.write(data)
        f.close()
    except Exception as ex:
        raise Exception (f"Error in save file:\n {ex}")



def replace_comma(fname: str):
    txt = read_text_file_line_by_line(fname=fname)
    txt = txt.replace(",","")
    save_text_file(fname=fname, data=txt)