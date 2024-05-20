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
            break
    return df_base

if __name__ == '__main__':

    root_path = "/home/saeed/Documents/Github/CatDB/Experiments/archive/Final-Results/"    
    
    columns = ["dataset_name", "config", "sub_task", "llm_model", "classifier", "task_type", "status",
                        "number_iteration","number_iteration_error", "has_description", "time_catalog_load", "time_pipeline_generate", "time_total", "time_execution", "train_accuracy", "train_f1_score", "train_log_loss", "train_r_squared", "train_rmse", "test_accuracy", "test_f1_score", "test_log_loss", "test_r_squared", "test_rmse"]
    
    df_catdb = load_results(f"{root_path}/Experiment1_LLM_Pipe_Gen_CatDB.dat")
    df_catdbchain = load_results(f"{root_path}/Experiment1_LLM_Pipe_Gen_CatDBChain.dat")
    
    df_pip_gen = pd.DataFrame(columns = columns)

    
    for index, row in df_catdb.iterrows():
        df_pip_gen.loc[len(df_pip_gen)] = row 

    for index, row in df_catdbchain.iterrows():
        df_pip_gen.loc[len(df_pip_gen)] = row 

    df_sort = pd.DataFrame(columns = columns)
    datasets = ["Balance-Scale","Breast-w","CMC","Credit-g","Diabetes","Tic-Tac-Toe",
                "Eucalyptus","PC1","Airlines","Jungle-Chess"]
    
    llms = ["gpt-4", "gpt-4o", "llama3-70b-8192"]

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
                    df_binary = df.loc[(df['train_accuracy'] >=0) &
                                (df['train_f1_score'] >=0) &
                                (df['test_accuracy'] >=0) &
                                (df['test_f1_score'] >=0) &
                                (df['task_type'] >='binary')]
                    df_binary = df_binary.sort_values(['train_accuracy','train_f1_score','test_accuracy', 'test_f1_score'], ascending=[False, False, False, False])
                    mergse_dfs(df_sort, df_binary)

                    # Multitask Tasks
                    df_multi = df.loc[(df['train_accuracy'] >=0) &
                                (df['train_log_loss'] >=0) &
                                (df['test_accuracy'] >=0) &
                                (df['test_log_loss'] >=0) &
                                (df['task_type'] >='multiclass')]
                    df_multi = df_multi.sort_values(['train_accuracy','train_log_loss','test_accuracy', 'test_log_loss'], ascending=[False, True, False, True])
                    mergse_dfs(df_sort, df_multi)


              

    df_sort.to_csv("/home/saeed/bb.csv", index=False)
   
           