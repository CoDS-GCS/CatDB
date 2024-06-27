import pandas as pd
from MergeResults import load_merge_all_results,load_results, get_top_k_binary, get_top_k_multiclass, get_top_k_regression, mergse_dfs 


if __name__ == '__main__':
    
    root_path = "/home/saeed/Documents/Github/CatDB/Experiments/archive/SIGMOD2025-Results/"        
    df_pip_gen = load_merge_all_results(root_path=root_path)
    df_sort = pd.DataFrame(columns = df_pip_gen.columns)
    
    df_cost = pd.DataFrame(columns = ["dataset_name","dataset_name_orig","config","tokens_count", "token_count_it1", "error_count","llm_model", "has_description"])

    df_csv_read = load_results(f"{root_path}/Experiment1_CSVDataReader.dat")
    df_exe = pd.DataFrame(columns = ["dataset_name","dataset_name_orig",
                                     "CatDB","CatDBChain","CAAFETabPFN","CAAFERandomForest","CatDB_min",
                                     "CatDBChain_min","CAAFETabPFN_min","CAAFERandomForest_min",
                                     "dataset_load_time", "llm_model", "has_description"])

    
    datasetIDs = [("Balance-Scale","oml_dataset_1_rnc",1),
                  ("Breast-w","oml_dataset_2_rnc",2),
                  ("CMC","oml_dataset_3_rnc",3),
                  ("Credit-g","oml_dataset_4_rnc",4),
                  ("Diabetes","oml_dataset_5_rnc",5),
                  ("Tic-Tac-Toe","oml_dataset_6_rnc",6),
                  ("Eucalyptus","oml_dataset_7_rnc",7),
                  ("PC1","oml_dataset_8_rnc",8),
                  ("Jungle-Chess","oml_dataset_10_rnc",9),                  
                  ("Higgs","oml_dataset_11_rnc",10),
                  ("Skin","oml_dataset_12_rnc",11),                  
                  ("Traffic","oml_dataset_19_rnc",12),
                  ("Walking-Activity","oml_dataset_20_rnc",13),
                  ("Black-Friday","oml_dataset_21_rnc",14),
                  ("Bike-Sharing","oml_dataset_22_rnc",15),
                  ("House-Sales","oml_dataset_23_rnc",16),
                  ("NYC","oml_dataset_24_rnc",17),
                  ("Airlines-DepDelay","oml_dataset_25_rnc",18)
                ]
    datasets = []
    for ds in datasetIDs:
        ds_name, ds_rnc_name, index = ds
        #datasets.append((ds_name, ds_name, index))
        datasets.append((ds_rnc_name, f"{ds_name}", index))

    llms = ["gemini-1.5-pro-latest","llama3-70b-8192"]
    classifier = ["Auto", "TabPFN", "RandomForest"]

    configs = ["CatDB", "CatDBChain", "CAAFE"]    
    ds_id = None
    for (ds,ds_title, index) in datasets:
        for llm in llms:
            for des in {"Yes", "No"}:  
                prompt_cost = {"CatDB": 0, "CatDBChain": 0, "CAAFETabPFN": 0, "CAAFERandomForest":0,
                               "CatDB_it1": 0, "CatDBChain_it1": 0, "CAAFETabPFN_it1": 0, "CAAFERandomForest_it1":0,
                               "CatDB_error_count": 0, "CatDBChain_error_count": 0}
                prompt_exe = {"CatDB": 0, "CatDBChain": 0, "CAAFETabPFN": 0, "CAAFERandomForest":0, 
                              "CatDB_min": 0.001, "CatDBChain_min": 0.001, "CAAFETabPFN_min": 0.001, "CAAFERandomForest_min":0.001}
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
                                df_binary = get_top_k_binary(df=df, config=config, k=10)
                                df_binary["number_iteration"] = [ki for ki in range(1, len(df_binary)+1)]
                                mergse_dfs(df_sort, df_binary)

                                # Multitask Tasks
                                df_multi = get_top_k_multiclass(df=df, config=config, k=10)
                                df_multi["number_iteration"] = [ki for ki in range(1, len(df_multi)+1)]
                                mergse_dfs(df_sort, df_multi)

                                # Regression Tasks
                                df_reg = get_top_k_regression(df=df, config=config, k=1)
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
                                        
                                fname = f"{fname}-{des}.csv"    
                                df_ds["ID"] =  [ds_ID for dsid in range(0,len(df_ds))]

                                if index <=10:
                                    df_ds.to_csv(f"{root_path}/seperate/{fname}", index=False)
                                
                                if config == "CatDB":
                                    prompt_cost[config] = df_ds['prompt_token_count'].sum()  
                                    prompt_cost["CatDB_it1"] = df_ds['all_token_count'].min() 
                                    prompt_cost["CatDB_error_count"] = df_ds['number_iteration_error'].min() 

                                    tmp_time = df_ds['time_total'].mean()
                                    prompt_exe[config] = f"{tmp_time:.2f}"  
                                    prompt_exe["CatDB_min"] = f"{tmp_time/60:.2f}"  

                                elif config == "CatDBChain":
                                    prompt_cost["CatDBChain"] = df_ds['prompt_token_count'].sum()  
                                    prompt_cost["CatDBChain_it1"] = df_ds['all_token_count'].min() 
                                    prompt_cost["CatDBChain_error_count"] = df_ds['number_iteration_error'].min() 

                                    tmp_time = (df_ds['time_total']+df_ds['time_pipeline_generate']).mean()
                                    prompt_exe[config] = f"{tmp_time:.2f}"  
                                    prompt_exe["CatDBChain_min"] = f"{tmp_time/60:.2f}"   

                                elif config == "CAAFE":
                                    prompt_cost[f"{config}{cls}_it1"] = df_ds.loc[df_ds['number_iteration'] == 1, 'all_token_count'].values[0]

                                    max_iteration = df_ds['number_iteration'].max()  
                                    tmp_time = df_ds['time_total'].mean()                 
                                    prompt_cost[f"{config}{cls}"] = df_ds.loc[df_ds['number_iteration'] == max_iteration, 'all_token_count'].values[0]
                                    prompt_exe[f"{config}{cls}"] = f"{tmp_time:.2f}"
                                    prompt_exe[f"{config}{cls}_min"] = f"{tmp_time/60:.2f}"
                                    
                if des == "No" and ds.endswith("rnc"):
                    df_cost.loc[len(df_cost)] = [ds, ds_title,"CatDB" ,prompt_cost["CatDB"], prompt_cost["CatDB_it1"],prompt_cost["CatDB_error_count"] ,llm, des]  
                    df_cost.loc[len(df_cost)] = [ds, ds_title,"CatDBChain" ,prompt_cost["CatDBChain"], prompt_cost["CatDBChain_it1"],prompt_cost["CatDBChain_error_count"],llm, des]
                    df_cost.loc[len(df_cost)] = [ds, ds_title,"CAAFETabPFN" ,prompt_cost["CAAFETabPFN"], prompt_cost["CAAFETabPFN_it1"],0,llm, des]
                    df_cost.loc[len(df_cost)] = [ds, ds_title,"CAAFERandomForest" ,prompt_cost["CAAFERandomForest"],prompt_cost["CAAFERandomForest_it1"],0,llm, des]

                   

                    dataset_load_time = df_csv_read.loc[df_csv_read['dataset']==ds]["time"].values[0] / 1000
                    df_exe.loc[len(df_cost)] = [ ds, ds_title, prompt_exe["CatDB"], prompt_exe["CatDBChain"], prompt_exe["CAAFETabPFN"],prompt_exe["CAAFERandomForest"],
                                                prompt_exe["CatDB_min"], prompt_exe["CatDBChain_min"], prompt_exe["CAAFETabPFN_min"],prompt_exe["CAAFERandomForest_min"], dataset_load_time, llm, des]      

              

    df_sort.to_csv(f"{root_path}/AllResults.csv", index=False)
    df_cost.to_csv(f"{root_path}/CostResults.csv", index=False)
    df_exe.to_csv(f"{root_path}/ExeResults.csv", index=False)           