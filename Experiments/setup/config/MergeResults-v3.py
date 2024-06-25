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

    results_path = [f"{root_path}/server-16/CatDB-gemini-1.5-orig/Experiment1_LLM_Pipe_Gen_CatDB.dat",
                    f"{root_path}/server-16/CatDB-gemini-1.5-orig/Experiment1_LLM_Pipe_Gen_CatDBChain.dat",
                    f"{root_path}/server-35/CatDB-gemini-1.5-rnc/Experiment1_LLM_Pipe_Gen_CatDB.dat",
                    f"{root_path}/server-35/CatDB-gemini-1.5-rnc/Experiment1_LLM_Pipe_Gen_CatDBChain.dat",
                    f"{root_path}/server-65/CAAFE-gemini-1.5/Experiment1_LLM_CAAFE.dat",
                    f"{root_path}/server-113/CatDB-llama3/Experiment1_LLM_Pipe_Gen_CatDB.dat",
                    f"{root_path}/server-113/CatDB-llama3/Experiment1_LLM_Pipe_Gen_CatDBChain.dat"] 
    
    df_pip_gen = pd.DataFrame(columns = columns)
    df_cost = pd.DataFrame(columns = ["dataset_name","dataset_name_orig","config","tokens_count","llm_model", "has_description"])

    df_csv_read = load_results(f"{root_path}/Experiment1_CSVDataReader.dat")
    df_exe = pd.DataFrame(columns = ["ID","dataset_name","dataset_name_orig","CatDB","CatDBChain","CAAFETabPFN","CAAFERandomForest", "dataset_load_time", "llm_model", "has_description"])

    for rp in results_path:
        df_catdb = load_results(rp)    
        df_pip_gen = merge_raw_data(df_result=df_pip_gen, df_tmp=df_catdb)

    df_sort = pd.DataFrame(columns = columns)
    
    datasetIDs = [("Balance-Scale","oml_dataset_1_rnc",7),
                  ("Breast-w","oml_dataset_2_rnc",1),
                  ("CMC","oml_dataset_3_rnc",8),
                  ("Credit-g","oml_dataset_4_rnc",2),
                  ("Diabetes","oml_dataset_5_rnc",3),
                  ("Tic-Tac-Toe","oml_dataset_6_rnc",4),
                  ("Eucalyptus","oml_dataset_7_rnc",9),
                  ("PC1","oml_dataset_8_rnc",5),
                  ("Airlines","oml_dataset_9_rnc",6),
                  ("Jungle-Chess","oml_dataset_10_rnc",10),
                  ("Adult","oml_dataset_26_rnc",11)
                ]
    # ,
    #               ("Higgs","oml_dataset_11_rnc",11),
    #               ("Skin","oml_dataset_12_rnc",12),
    #               ("Click-Prediction","oml_dataset_13_rnc",13),
    #               ("Census-Augmented","oml_dataset_14_rnc",14),
    #               ("Heart-Statlog","oml_dataset_15_rnc",15),
    #               ("KDDCup99","oml_dataset_16_rnc",16),
    #               ("Road-Safety","oml_dataset_17_rnc",17),
    #               ("Drug-Directory","oml_dataset_18_rnc",18),
    #               ("Traffic","oml_dataset_19_rnc",19),
    #               ("Walking-Activity","oml_dataset_20_rnc",20),
    #               ("Black-Friday","oml_dataset_21_rnc",21),
    #               ("Bike-Sharing","oml_dataset_22_rnc",22),
    #               ("House-Sales","oml_dataset_23_rnc",23),
    #               ("NYC","oml_dataset_24_rnc",24),
    #               ("Airlines-DepDelay","oml_dataset_25_rnc",25),
    #               ("Adult","oml_dataset_26_rnc",26)

    datasets = []
    for ds in datasetIDs:
        ds_name, ds_rnc_name, ds_id = ds
        datasets.append((ds_name, ds_name, ds_id))
        datasets.append((ds_rnc_name, f"{ds_name}-rnc", ds_id))

    #llms = ["gpt-4", "gpt-4o", "llama3-70b-8192", "gemini-1.5-pro-latest"]
    llms = ["gemini-1.5-pro-latest","llama3-70b-8192"]
    classifier = ["Auto", "TabPFN", "RandomForest"]

    configs = ["CatDB", "CatDBChain", "CAAFE"]    

    for (ds,ds_title,ds_id) in datasets:
        for llm in llms:
            for des in {"Yes", "No"}:  
                prompt_cost = {"CatDB": 0, "CatDBChain": 0, "CAAFETabPFN": 0, "CAAFERandomForest":0}
                prompt_exe = {"CatDB": 0, "CatDBChain": 0, "CAAFETabPFN": 0, "CAAFERandomForest":0}
                for config in configs:
                    for cls in classifier:                    
                            df = df_pip_gen.loc[(df_pip_gen['dataset_name'] == ds) & 
                                                (df_pip_gen['config'] == config) &
                                                (df_pip_gen['llm_model'] == llm) &
                                                (df_pip_gen['status'] == True) &
                                                (df_pip_gen['classifier'] == cls) &
                                                (df_pip_gen['has_description'] == des) &
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
                                                    (df_sort['classifier'] == cls) &
                                                    (df_sort['config'] == config)]
                                #ds_ID = config          
                                
                                df_ds["dataset_name_orig"] = [ds_title for myi in range(0, len(df_ds))]

                                fname = f"{config}-{ds_title}-{llm}"
                                if cls != "Auto":
                                    fname = f"{fname}-{cls}"
                                    #ds_ID = f"{config}-{cls}"

                                if config == "CatDB":
                                    ds_ID = 1
                                elif config == "CatDBChain":
                                    ds_ID = 2    

                                elif cls == "TabPFN":
                                    ds_ID = 3
                                elif cls == "RandomForest":
                                    ds_ID = 4                                           
                                        
                                fname = f"{fname}-{des}.csv"    
                                df_ds["ID"] =  [ds_ID for dsid in range(0,len(df_ds))]

                                df_ds.to_csv(f"{root_path}/seperate/{fname}", index=False)
                                
                                if config == "CatDB":
                                    prompt_cost[config] = df_ds['prompt_token_count'].sum()   
                                    prompt_exe[config] = f"{df_ds['time_total'].mean():.2f}"  

                                elif config == "CatDBChain":
                                    prompt_cost[config] = df_ds['prompt_token_count'].sum()   
                                    prompt_exe[config] = f"{(df_ds['time_total']+df_ds['time_pipeline_generate']).mean():.2f}"    

                                elif config == "CAAFE":
                                    max_iteration = df_ds['number_iteration'].max()                    
                                    prompt_cost[f"{config}{cls}"] = df_ds.loc[df_ds['number_iteration'] == max_iteration, 'all_token_count'].values[0]
                                    prompt_exe[f"{config}{cls}"] = f"{df_ds['time_total'].mean():.2f}"
                                    
                if des == "No" and ds.endswith("rnc"):
                    df_cost.loc[len(df_cost)] = [ds, ds_title,"CatDB" ,prompt_cost["CatDB"],llm, des]  
                    df_cost.loc[len(df_cost)] = [ds, ds_title,"CatDBChain" ,prompt_cost["CatDBChain"],llm, des]
                    df_cost.loc[len(df_cost)] = [ds, ds_title,"CAAFETabPFN" ,prompt_cost["CAAFETabPFN"],llm, des]
                    df_cost.loc[len(df_cost)] = [ds, ds_title,"CAAFERandomForest" ,prompt_cost["CAAFERandomForest"],llm, des]


                    dataset_load_time = df_csv_read.loc[df_csv_read['dataset']==ds]["time"].values[0] / 1000
                    df_exe.loc[len(df_cost)] = [ds_id, ds, ds_title, prompt_exe["CatDB"], prompt_exe["CatDBChain"], prompt_exe["CAAFETabPFN"],prompt_exe["CAAFERandomForest"], dataset_load_time, llm, des]      

              

    df_sort.to_csv(f"{root_path}/AllResults.csv", index=False)
    df_cost.to_csv(f"{root_path}/CostResults.csv", index=False)
    df_exe.to_csv(f"{root_path}/ExeResults.csv", index=False)
           