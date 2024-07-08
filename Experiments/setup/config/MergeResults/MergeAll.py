import pandas as pd
from MergeResults import load_merge_all_results,load_results, get_top_k_binary, get_top_k_multiclass, get_top_k_regression, mergse_dfs, get_top_k_chain 


if __name__ == '__main__':
    
    root_path = "/home/saeed/Documents/Github/CatDB/Experiments/archive/SIGMOD2025-Results/"        
    df_pip_gen = load_merge_all_results(root_path=root_path)
    df_sort = pd.DataFrame(columns = df_pip_gen.columns)
    
    df_cost = pd.DataFrame(columns = ["dataset_name","dataset_name_orig","config",
                                      "tokens_count","pp_tokens_count","fe_tokens_count",                                      
                                      "token_count_it1","pp_token_count_it1","fe_token_count_it1",
                                      "token_count_err_it1","pp_token_count_err_it1","fe_token_count_err_it1",
                                      "error_count", "pp_error_count","fe_error_count",
                                      "error_count_it1", "pp_error_count_it1","fe_error_count_it1",
                                      "llm_model", "has_description", "task_type","task","samples"])

    df_csv_read = load_results(f"{root_path}/Experiment1_CSVDataReader.dat")
    df_exe = pd.DataFrame(columns = ["dataset_name","dataset_name_orig",
                                     "CatDB","CatDBChain","CAAFETabPFN","CAAFERandomForest","CatDB_min",
                                     "CatDBChain_min","CAAFETabPFN_min","CAAFERandomForest_min",
                                     "dataset_load_time", "llm_model", "has_description", "task_type","task","samples"])    
   

    
    datasetIDs = [("Balance-Scale","oml_dataset_1_rnc","multiclass",1),
                  ("Breast-w","oml_dataset_2_rnc","binary",2),
                  ("CMC","oml_dataset_3_rnc","multiclass",3),
                  ("Credit-g","oml_dataset_4_rnc","binary",4),
                  ("Diabetes","oml_dataset_5_rnc","binary",5),
                  ("Tic-Tac-Toe","oml_dataset_6_rnc","binary",6),
                  ("Eucalyptus","oml_dataset_7_rnc","multiclass",7),
                  ("PC1","oml_dataset_8_rnc","binary",8),
                  ("Jungle-Chess","oml_dataset_10_rnc","multiclass",9),                  
                  ("Higgs","oml_dataset_11_rnc","binary",10),
                  ("Skin","oml_dataset_12_rnc","binary",11),                  
                  ("Traffic","oml_dataset_19_rnc","multiclass",12),
                  ("Walking-Activity","oml_dataset_20_rnc","multiclass",13),
                  ("Black-Friday","oml_dataset_21_rnc","regression",14),
                  ("Bike-Sharing","oml_dataset_22_rnc","regression",15),
                  ("House-Sales","oml_dataset_23_rnc","regression",16),
                  ("NYC","oml_dataset_24_rnc","regression",17),
                  ("Airlines-DepDelay","oml_dataset_25_rnc","regression",18)
                ]
    
    datasets = []
    for ds in datasetIDs:
        ds_name, ds_rnc_name, task_type, index = ds
        #datasets.append((ds_name, ds_name, index))
        datasets.append((ds_rnc_name, f"{ds_name}",task_type , index))

    llms = ["gemini-1.5-pro-latest","llama3-70b-8192", "gpt-4o"]
    classifier = ["Auto", "TabPFN", "RandomForest"]

    configs = ["CatDB", "CatDBChain", "CAAFE"]    
    ds_id = None
    for samples in {0}:
        for (ds,ds_title, task_type, index) in datasets:
            for llm in llms:
                for des in {"Yes", "No"}:  
                    prompt_cost = dict()               
                    for ck in df_cost.columns:
                        if ck == "dataset_name":
                            prompt_cost[ck] = ds
                        elif ck == "dataset_name_orig":
                            prompt_cost[ck] = ds_title   
                        elif ck in {"config", "llm_model", "has_description"}:
                            prompt_cost[ck] = None  
                        else:
                            for mycls in {"CatDB", "CatDBChain", "CAAFETabPFN", "CAAFERandomForest"}:
                                prompt_cost[f"{mycls}{ck}"] = 0
                    
                    prompt_exe = {"CatDB": 0, "CatDBChain": 0, "CAAFETabPFN": 0, "CAAFERandomForest":0, 
                                "CatDB_min": 0.001, "CatDBChain_min": 0.001, "CAAFETabPFN_min": 0.001, "CAAFERandomForest_min":0.001}
                    for config in configs:
                        for cls in classifier: 
                                tmp_df = df_pip_gen.loc[(df_pip_gen['dataset_name'] == ds) & 
                                                    (df_pip_gen['config'] == config) &
                                                    (df_pip_gen['llm_model'] == llm) &
                                                    (df_pip_gen['status'] == True) &
                                                    (df_pip_gen['classifier'] == cls) &
                                                    (df_pip_gen['has_description'] == des) &
                                                    (df_pip_gen['number_of_samples'] == samples)]
                                                
                                df_chain = tmp_df.loc[(tmp_df['sub_task'] == 'DataPreprocessing') | 
                                                (tmp_df['sub_task'] == 'FeatureEngineering')]
                                
                                df = tmp_df.loc[(tmp_df['sub_task'] == 'ModelSelection') | (pd.isnull(tmp_df['sub_task']))]

                                if len(df) > 0:
                                    
                                    # Binary Tasks
                                    df_binary = get_top_k_binary(df=df, config=config, k=10)
                                    df_binary["number_iteration"] = [ki for ki in range(1, len(df_binary)+1)]
                                    mergse_dfs(df_sort, df_binary)

                                    # Multitask Tasks
                                    df_multi = get_top_k_multiclass(df=df, config=config, k=10)
                                    df_multi["number_iteration"] = [ki for ki in range(1, len(df_multi)+1)]
                                    mergse_dfs(df_sort, df_multi)

                                    # Regression Tasks
                                    df_reg = get_top_k_regression(df=df, config=config, k=10)
                                    df_reg["number_iteration"] = [ki for ki in range(1, len(df_reg)+1)]
                                    mergse_dfs(df_sort, df_reg)

                                    df_ds = df_sort.loc[(df_sort['dataset_name'] ==ds) &
                                                        (df_sort['llm_model'] == llm)&
                                                        (df_sort['classifier'] == cls) &
                                                        (df_sort['config'] == config)]    
                                    
                                    df_ds["dataset_name_orig"] = [ds_title for myi in range(0, len(df_ds))]

                                    fname = f"{config}-{ds_title}-{llm}"
                                    if cls != "Auto":
                                        fname = f"{fname}-{cls}"

                                    if config == "CatDB":
                                        ds_ID = 1
                                    elif config == "CatDBChain":
                                        ds_ID = 2    

                                    elif cls == "TabPFN":
                                        ds_ID = 3
                                    elif cls == "RandomForest":
                                        ds_ID = 4                                           
                                            
                                    fname = f"{fname}-{samples}-{des}.csv"    
                                    df_ds["ID"] =  [ds_ID for dsid in range(0,len(df_ds))]

                                    if index <=10:
                                        df_ds.to_csv(f"{root_path}/seperate/{fname}", index=False)                        
                                    
                                    if task_type in {"binary", "multiclass"}:
                                        tsk = "classification"
                                    else:
                                        tsk = "regression"

                                    cindex = len(df_cost)
                                    row_entry = [ds, ds_title, config]
                                    for ic in range(0,len(df_cost.columns)-8):
                                        row_entry.append(0)
                                    row_entry.append(llm)
                                    row_entry.append(des)    
                                    row_entry.append(task_type) 
                                    row_entry.append(tsk) 
                                    row_entry.append(samples)  
                                    df_cost.loc[cindex] = row_entry 
                                                                
                                    if config == "CatDB":
                                        df_cost.at[cindex,"config"]="CatDB"
                                        df_cost.at[cindex,"tokens_count"] = df_ds['prompt_token_count'].sum()  

                                        df_ds_tmp = df_ds.sort_values(by='all_token_count', ascending=True).reset_index(drop=True).head(1)

                                        df_cost.at[cindex,"token_count_it1"] = df_ds_tmp.loc[0, 'prompt_token_count']
                                        df_cost.at[cindex,"token_count_err_it1"] = df_ds_tmp.loc[0,'all_token_count'] - df_ds_tmp.loc[0, 'prompt_token_count']
                                        df_cost.at[cindex,"error_count"] = df_ds_tmp.loc[0,'number_iteration_error']-1

                                        tmp_time = df_ds['time_total'].mean()
                                        if tmp_time <= 15:
                                            tmp_time=15 
                                        prompt_exe[config] = f"{tmp_time:.2f}"  
                                        prompt_exe["CatDB_min"] = f"{tmp_time/60:.2f}"  

                                    elif config == "CatDBChain":
                                        df_cost.at[cindex,"config"]="CatDBChain"
                                        df_chain_pp = get_top_k_chain(df=df_chain, sub_task="DataPreprocessing", k=1)
                                        df_chain_fe = get_top_k_chain(df=df_chain, sub_task="FeatureEngineering", k=1)

                                        avg_pp_token = df_chain_pp.loc[0, 'prompt_token_count']
                                        avg_fe_token = df_chain_fe.loc[0, 'prompt_token_count']

                                        avg_pp_err_token = df_chain_pp.loc[0, 'all_token_count'] - avg_pp_token
                                        avg_fe_err_token = df_chain_fe.loc[0, 'all_token_count'] - avg_fe_token

                                        avg_pp_error = df_chain_pp.loc[0, 'number_iteration_error'] -1
                                        avg_fe_error = df_chain_fe.loc[0, 'number_iteration_error'] -1

                                        df_cost.at[cindex,"tokens_count"] = int(df_ds['prompt_token_count'].sum() + len(df_ds) * (avg_pp_token * 0.7 + avg_fe_token*0.7))

                                        df_ds_tmp = df_ds.sort_values(by='all_token_count', ascending=True).reset_index(drop=True).head(1)

                                        df_cost.at[cindex,"pp_tokens_count"] = int(len(df_ds) * (avg_pp_token * 0.7 +  avg_pp_err_token))
                                        df_cost.at[cindex,"fe_tokens_count"] = int(len(df_ds) * (avg_fe_token * 0.7 + avg_fe_err_token))

                                        df_cost.at[cindex,"error_count"] = df_ds_tmp.loc[0, 'number_iteration_error'] - 1 +  avg_pp_error +  avg_fe_error
                                        df_cost.at[cindex,"token_count_it1"] = df_ds_tmp.loc[0, 'prompt_token_count']
                                        df_cost.at[cindex,"token_count_err_it1"] = df_ds_tmp.loc[0,'all_token_count'] - df_ds_tmp.loc[0, 'prompt_token_count']
                                        df_cost.at[cindex,"pp_token_count_it1"] = int(avg_pp_token * 0.7)
                                        df_cost.at[cindex,"fe_token_count_it1"] = int(avg_fe_token * 0.7)
                                        df_cost.at[cindex,"pp_token_count_err_it1"] = avg_pp_err_token
                                        df_cost.at[cindex,"fe_token_count_err_it1"] = avg_fe_err_token

                                        df_cost.at[cindex,"pp_error_count"] = avg_pp_error
                                        df_cost.at[cindex,"fe_error_count"] = avg_fe_error

                                        tmp_time = (df_ds['time_total']+df_ds['time_pipeline_generate']).mean()
                                        if tmp_time <= 15:
                                            tmp_time=15 
                                        prompt_exe[config] = f"{tmp_time:.2f}"  
                                        prompt_exe["CatDBChain_min"] = f"{tmp_time/60:.2f}"   

                                    elif config == "CAAFE":
                                        df_cost.at[cindex,"config"]=f"{config}{cls}"
                                        df_cost.at[cindex,"token_count_it1"] = df_ds.loc[df_ds['number_iteration'] == 1, 'all_token_count'].values[0]

                                        max_iteration = df_ds['number_iteration'].max()  
                                        tmp_time = df_ds['time_total'].mean() 

                                        # missed iterations
                                        missed_count = 10 - max_iteration
                                        missed_tokens = 0
                                        if missed_count > 0:
                                            base_prompt = df_ds.loc[df_ds['number_iteration'] == max_iteration, 'prompt_token_count'].values[0]
                                            missed_tokens = (missed_count +1)* base_prompt                                            
                                                    
                                        df_cost.at[cindex,"tokens_count"] = df_ds.loc[df_ds['number_iteration'] == max_iteration, 'all_token_count'].values[0] + missed_tokens
                                        prompt_exe[f"{config}{cls}"] = f"{tmp_time:.2f}"
                                        prompt_exe[f"{config}{cls}_min"] = f"{tmp_time/60:.2f}"
                                        
                    if des == "No" and ds.endswith("rnc"):    
                        if task_type in {"binary", "multiclass"}:
                            tsk = "classification"
                        else:
                            tsk = "regression"                  

                        dataset_load_time = df_csv_read.loc[df_csv_read['dataset']==ds]["time"].values[0] / 1000
                        df_exe.loc[len(df_exe)] = [ ds, ds_title, prompt_exe["CatDB"], prompt_exe["CatDBChain"], prompt_exe["CAAFETabPFN"],prompt_exe["CAAFERandomForest"],
                                                    prompt_exe["CatDB_min"], prompt_exe["CatDBChain_min"], prompt_exe["CAAFETabPFN_min"],prompt_exe["CAAFERandomForest_min"], dataset_load_time, llm, des, task_type, tsk, samples]      

              

    df_sort.to_csv(f"{root_path}/AllResults.csv", index=False)
    df_cost.to_csv(f"{root_path}/CostResults.csv", index=False)
    df_exe.to_csv(f"{root_path}/ExeResults.csv", index=False)           