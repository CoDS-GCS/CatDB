import os
import sys
import pandas as pd
import numpy as np

def load_results(path):
    df = pd.read_csv(path)
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


if __name__ == '__main__':

    root_path = "/home/saeed/Downloads/tmp/"    
    
    columns = ["dataset_name", "config", "sub_task", "llm_model", "classifier", "task_type", "status",
                        "number_iteration","number_iteration_error", "has_description", "time_catalog_load", "time_pipeline_generate",
                        "time_total", "time_execution", "train_auc","train_auc_ovo","train_auc_ovr", "train_accuracy",
                        "train_f1_score", "train_log_loss", "train_r_squared", "train_rmse", "test_auc","test_auc_ovo",
                        "test_auc_ovr", "test_accuracy", "test_f1_score", "test_log_loss", "test_r_squared", "test_rmse",
                        "prompt_token_count","all_token_count", "operation"]
    
    results_path = [f"{root_path}/Experiment1_LLM_Pipe_Gen_CatDB_1.dat",
                    f"{root_path}/Experiment1_LLM_Pipe_Gen_CatDB_2.dat",
                    f"{root_path}/Experiment1_LLM_Pipe_Gen_CatDBChain_1.dat",
                    f"{root_path}/Experiment1_LLM_Pipe_Gen_CatDBChain_2.dat"] 
    
    df_pip_gen = pd.DataFrame(columns = columns)

    for rp in results_path:
        df_catdb = load_results(rp)    
        df_pip_gen = merge_raw_data(df_result=df_pip_gen, df_tmp=df_catdb)

    df_sort = pd.DataFrame(columns = columns)
    datasets = ["Balance-Scale","Breast-w","CMC","Credit-g","Diabetes","Tic-Tac-Toe",
                "Eucalyptus","PC1","Airlines","Jungle-Chess"]
    
    llms = ["gpt-4", "gpt-4o", "llama3-70b-8192", "gemini-1.5-pro-latest"]

    configs = ["CatDB", "CatDBChain"]
    

    for ds in datasets:
        for llm in llms:
            for config in configs:
                #print(f"{config} >> {len(df_pip_gen[]['sub_task'])}")
                df = df_pip_gen.loc[(df_pip_gen['dataset_name'] == ds) & 
                                    (df_pip_gen['config'] == config) &
                                    (df_pip_gen['llm_model'] == llm) &
                                    (df_pip_gen['status'] == True) &
                                    ((df_pip_gen['sub_task'] == 'ModelSelection') | (pd.isnull(df_pip_gen['sub_task'])))]
                
                
                if len(df) > 0:
                    
                    # Binary Tasks
                    df_binary = df.loc[(df['train_auc'] >=0) &
                                (df['test_auc'] >=0) &
                                (df['task_type'] =='binary')]
                    df_binary = df_binary.head(10)
                    df_binary["number_iteration"] = [ki for ki in range(1, len(df_binary)+1)]
                    mergse_dfs(df_sort, df_binary)

                    # Multitask Tasks
                    df_multi = df.loc[(df['train_auc_ovr'] >=0) &
                                (df['test_auc_ovr'] >=0) &
                                (df['task_type'] =='multiclass')]
                    df_multi = df_multi.head(10)
                    df_multi["number_iteration"] = [ki for ki in range(1, len(df_multi)+1)]
                    mergse_dfs(df_sort, df_multi)


              

    df_sort.to_csv("/home/saeed/cc.csv", index=False)
   
           