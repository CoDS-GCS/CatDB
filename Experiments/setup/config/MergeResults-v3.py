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

    root_path = "/home/saeed/Documents/Github/CatDB/Experiments/archive/SIGMOD2025-Results/"    
    
    columns = ["dataset_name", "config", "sub_task", "llm_model", "classifier", "task_type", "status",
                        "number_iteration","number_iteration_error", "has_description", "time_catalog_load", "time_pipeline_generate",
                        "time_total", "time_execution", "train_auc","train_auc_ovo","train_auc_ovr", "train_accuracy",
                        "train_f1_score", "train_log_loss", "train_r_squared", "train_rmse", "test_auc","test_auc_ovo",
                        "test_auc_ovr", "test_accuracy", "test_f1_score", "test_log_loss", "test_r_squared", "test_rmse",
                        "prompt_token_count","all_token_count", "operation"]

    results_path = [f"{root_path}/server-16/part1-CatDB-gemini-1.5-pro-latest/Experiment1_LLM_Pipe_Gen_CatDB.dat",
                    f"{root_path}/server-35/part1-CatDB-mixtral-llama/Experiment1_LLM_Pipe_Gen_CatDB.dat",
                    f"{root_path}/server-65/part1-CatDB-gemini-1.5-pro-latest/Experiment1_LLM_Pipe_Gen_CatDB.dat",
                    f"{root_path}/server-65/part1-CatDB-gemini-1.5-pro-latest/Experiment1_LLM_Pipe_Gen_CatDBChain.dat",
                    f"{root_path}/server-65/part2-CAFFE-gemini-1.5-pro-latest/Experiment1_LLM_CAAFE.dat"] 
    
    df_pip_gen = pd.DataFrame(columns = columns)

    for rp in results_path:
        df_catdb = load_results(rp)    
        df_pip_gen = merge_raw_data(df_result=df_pip_gen, df_tmp=df_catdb)

    df_sort = pd.DataFrame(columns = columns)
    datasets = [("Balance-Scale",1), ("CMC",2), ("Eucalyptus",3), ("Jungle-Chess",4),
                ("Breast-w",1), ("Credit-g",2), ("Diabetes",3), ("Tic-Tac-Toe",4),
                ("PC1",5), ("Airlines",6)]

    
    #llms = ["gpt-4", "gpt-4o", "llama3-70b-8192", "gemini-1.5-pro-latest"]
    llms = ["gemini-1.5-pro-latest"]
    classifier = ["Auto", "TabPFN", "RandomForest"]

    configs = ["CatDB", "CatDBChain", "CAAFE"]
    

    for (ds,ds_id) in datasets:
        for llm in llms:
            for config in configs:
                for cls in classifier:
                    #print(f"{config} >> {len(df_pip_gen[]['sub_task'])}")
                    df = df_pip_gen.loc[(df_pip_gen['dataset_name'] == ds) & 
                                        (df_pip_gen['config'] == config) &
                                        (df_pip_gen['llm_model'] == llm) &
                                        (df_pip_gen['status'] == True) &
                                        (df_pip_gen['classifier'] == cls) &
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

                        df_ds = df_sort.loc[(df_sort['dataset_name'] ==ds) &
                                            (df_sort['llm_model'] == llm)&
                                            (df_sort['config'] == config)]
                        df_ds["ID"] = [ds_id for dsid in range(0,len(df_ds))]

                        fname = f"{config}-{ds}-{llm}"
                        if cls != "Auto":
                            fname = f"{fname}-{cls}"
                        fname = f"{fname}.csv"    

                        df_ds.to_csv(f"/home/saeed/Documents/Github/CatDB/Experiments/archive/SIGMOD2025-Results/seperate/{fname}", index=False)
                    


              

    df_sort.to_csv("/home/saeed/Documents/Github/CatDB/Experiments/archive/SIGMOD2025-Results/AllResults.csv", index=False)
   
           