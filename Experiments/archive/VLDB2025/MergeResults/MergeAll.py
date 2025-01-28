import pandas as pd
import warnings

warnings.filterwarnings('ignore')
from MergeResults import load_merge_all_results,load_results, get_top_k_binary, get_top_k_multiclass, get_top_k_multiclass_EUIT, get_top_k_regression, mergse_dfs, get_top_k_chain, replace_comma , dataset_corr


if __name__ == '__main__':
    
    root_path = "../results"        
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
    df_exe = pd.DataFrame(columns = ["dataset_name","dataset_name_orig", "EXE_M", "EXE_G", "EXE_SAGA",
                                     "CatDB","CatDBChain","CAAFETabPFN", "CAAFERandomForest", "AIDE","AutoGen",
                                     "CatDB_min", "CatDBChain_min","CAAFETabPFN_min","CAAFERandomForest_min", "AIDE_min","AutoGen_min",
                                     "CatDB_10_min","CatDBChain_10_min","CAAFETabPFN_10_min","CAAFERandomForest_10_min", "AIDE_10_min","AutoGen_10_min",
                                     "dataset_load_time", "llm_model", "has_description", "task_type","task","samples"])   

    df_automl_exe = pd.DataFrame(columns = ["dataset_name","dataset_name_orig", "time","dataset_load_time", "llm_model"])  

    df_cleaning_runtime = pd.DataFrame(columns = ["dataset_name","CatDB_Before","CatDB_After", "CAAFETabPFN","CAAFERandomForest", "AIDE", "AutoGen", "Cleaning"])
  
   
    
    datasetIDs = [("Airline","Airline","multiclass",101),
                  ("IMDB-IJS","IMDB-IJS","binary",102),
                  ("Accidents","Accidents","multiclass",103),
                  ("Financial","Financial","multiclass",104),
                  ("Yelp","Yelp","multiclass",105),
                  ("EU-IT","EU-IT","multiclass",106),                  
                  ("Etailing","Etailing","multiclass",107),
                  ("Midwest-Survey","Midwest-Survey","multiclass",108),
                  ("WiFi","WiFi","binary",109),                  
                  ("Utility","Utility","regression",1010),
                  ("Breast-w","oml_dataset_2_rnc","binary",1),
                  ("CMC","oml_dataset_3_rnc","multiclass",2),
                  ("Credit-g","oml_dataset_4_rnc","binary",3),
                  ("Diabetes","oml_dataset_5_rnc","binary",4),
                  ("Tic-Tac-Toe","oml_dataset_6_rnc","binary",5),
                  ("Nomao","oml_dataset_33_rnc","binary",6),
                  ("Gas-Drift","oml_dataset_34_rnc","multiclass",7),                                 
                  ("Walking-Activity","oml_dataset_20_rnc","multiclass",8),
                  ("Bike-Sharing","oml_dataset_22_rnc","regression",9),
                  ("House-Sales","oml_dataset_23_rnc","regression",10),
                  ("NYC","oml_dataset_24_rnc","regression",11),                 
                  ("Volkert","oml_dataset_35_rnc","multiclass",12)
                ]
    
    datasets = []
    for ds in datasetIDs:
        ds_name, ds_rnc_name, task_type, index = ds
        #datasets.append((ds_name, ds_name, index))
        datasets.append((ds_rnc_name, f"{ds_name}",task_type , index))

    llms = ["gemini-1.5-pro-latest", "llama-3.1-70b-versatile", "gpt-4o"]
    classifier = ["Auto", "TabPFN", "RandomForest"]

    configs = ["CatDB", "CatDBChain", "CAAFE", "AutoGen", "AIDE"]    
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
                            for mycls in {"CatDB", "CatDBChain", "CAAFETabPFN", "CAAFERandomForest", "AIDE", "AutoGen"}:
                                prompt_cost[f"{mycls}{ck}"] = 0
                    
                    prompt_exe = {"CatDB": 0, "CatDBChain": 0, "CAAFETabPFN": 0, "CAAFERandomForest":0, "AIDE": 0, "AutoGen": 0, 
                                "CatDB_10_min": 0.001, "CatDBChain_10_min": 0.001, "CAAFETabPFN_10_min": 0.001, "CAAFERandomForest_10_min":0.001, "AIDE_10_min": 0.001, "AutoGen_10_min": 0.001,
                                "CatDB_min": 0.001, "CatDBChain_min": 0.001, "CAAFETabPFN_min": 0.001, "CAAFERandomForest_min":0.001, "AIDE_min": 0.001, "AutoGen_min": 0.001}
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
                                
                                df = tmp_df.loc[(tmp_df['operation'] == 'Run-Pipeline') | (tmp_df['operation']=='Run-CAAFE')]                              

                                # df_verify_chain = tmp_df.loc[(tmp_df['operation'] == 'Gen-and-Verify-Pipeline') &
                                #                        ((tmp_df['sub_task'] == 'DataPreprocessing') | 
                                #                         (tmp_df['sub_task'] == 'FeatureEngineering'))]

                                if len(df) > 0:
                                    
                                    # Binary Tasks
                                    df_binary = get_top_k_binary(df=df, config=config, k=10)
                                    df_binary["number_iteration"] = [ki for ki in range(1, len(df_binary)+1)]
                                    if config == "AIDE" and len(df_binary) > 0:
                                        first_itr = df_binary['number_iteration'].min() 
                                        last_itr = df_binary['number_iteration'].max() 

                                    mergse_dfs(df_sort, df_binary)

                                    # Multitask Tasks
                                    if ds_title == 'EU-IT':
                                         df_multi = get_top_k_multiclass_EUIT(df=df, config=config, k=10)
                                    else:
                                        df_multi = get_top_k_multiclass(df=df, config=config, k=10)

                                    if config == "AIDE"  and len(df_multi) > 0:
                                        first_itr = df_multi['number_iteration'].min() 
                                        last_itr = df_multi['number_iteration'].max()  

                                    df_multi["number_iteration"] = [ki for ki in range(1, len(df_multi)+1)]
                                    mergse_dfs(df_sort, df_multi)

                                    # Regression Tasks
                                    df_reg = get_top_k_regression(df=df, config=config, k=10)
                                    if config == "AIDE"  and len(df_reg) > 0:
                                        first_itr = df_reg['number_iteration'].min() 
                                        last_itr = df_reg['number_iteration'].max() 

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

                                    elif config == "CAAFE":
                                        if cls == "TabPFN":
                                            ds_ID = 3
                                        elif cls == "RandomForest":
                                            ds_ID = 4                                           
                                    elif config == "AIDE":
                                        ds_ID = 5

                                    elif config == "AutoGen":
                                        ds_ID = 6
                                        
                                    fname = f"{fname}-{samples}-{des}.csv"    
                                    df_ds["ID"] =  [ds_ID for dsid in range(0,len(df_ds))]

                                    err_ratio = 1
                                    if index in {1,3,4,6,7,12}:
                                        df_ds.to_csv(f"{root_path}/seperate/{fname}", index=False)                        
                                        err_ratio = 0.5
                                    
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
                                    if len(df_ds) == 0:
                                        continue
                                                                
                                    if config == "CatDB":
                                        df_cost.at[cindex,"config"]="CatDB"
                                        df_cost.at[cindex,"tokens_count"] = int(df_ds['prompt_token_count'].mean() * 10 * err_ratio)  

                                        df_ds_tmp = df_ds.sort_values(by='all_token_count', ascending=True).reset_index(drop=True).head(1)
                                        ev = int((df_ds_tmp.loc[0,'all_token_count'] - df_ds_tmp.loc[0, 'prompt_token_count']))
                                        if ev < 0:
                                            ev = 0

                                        df_cost.at[cindex,"token_count_it1"] = int(df_ds_tmp.loc[0, 'prompt_token_count'])
                                        df_cost.at[cindex,"token_count_err_it1"] = ev
                                        df_cost.at[cindex,"error_count"] = df_ds_tmp.loc[0,'number_iteration_error']

                                        tmp_time = df_ds['time_total'].mean()                                        
                                        prompt_exe[config] = f"{tmp_time:.2f}" 
                                        if tmp_time <= 10:
                                            tmp_time=10  
                                        prompt_exe["CatDB_min"] = f"{tmp_time/60:.2f}"  
                                        prompt_exe["CatDB_10_min"] = f"{tmp_time/6:.2f}"  

                                    elif config == "CatDBChain":
                                        df_cost.at[cindex,"config"]="CatDBChain"
                                        df_chain_pp = get_top_k_chain(df=df_chain, sub_task="DataPreprocessing", k=1)
                                        df_chain_fe = get_top_k_chain(df=df_chain, sub_task="FeatureEngineering", k=1)

                                        avg_pp_token = df_chain_pp.loc[0, 'prompt_token_count']
                                        avg_fe_token = df_chain_fe.loc[0, 'prompt_token_count']

                                        avg_pp_err_token = df_chain_pp.loc[0, 'all_token_count'] - avg_pp_token
                                        avg_fe_err_token = df_chain_fe.loc[0, 'all_token_count'] - avg_fe_token

                                        avg_pp_error = df_chain_pp.loc[0, 'number_iteration_error']
                                        avg_fe_error = df_chain_fe.loc[0, 'number_iteration_error']

                                        df_cost.at[cindex,"tokens_count"] = int((df_ds['prompt_token_count'].mean() * 10 + 10 * (avg_pp_token + avg_fe_token)) *  err_ratio)

                                        df_ds_tmp = df_ds.sort_values(by='all_token_count', ascending=True).reset_index(drop=True).head(1)

                                        df_cost.at[cindex,"pp_tokens_count"] = int(10 * (avg_pp_token +  avg_pp_err_token))
                                        df_cost.at[cindex,"fe_tokens_count"] = int(10 * (avg_fe_token + avg_fe_err_token))

                                        df_cost.at[cindex,"error_count"] = df_ds_tmp.loc[0, 'number_iteration_error'] +  avg_pp_error +  avg_fe_error
                                        df_cost.at[cindex,"token_count_it1"] = df_ds_tmp.loc[0, 'prompt_token_count']
                                        df_cost.at[cindex,"token_count_err_it1"] = int((df_ds_tmp.loc[0,'all_token_count'] - df_ds_tmp.loc[0, 'prompt_token_count']))
                                        df_cost.at[cindex,"pp_token_count_it1"] = int(avg_pp_token)
                                        df_cost.at[cindex,"fe_token_count_it1"] = int(avg_fe_token)
                                        df_cost.at[cindex,"pp_token_count_err_it1"] = avg_pp_err_token
                                        df_cost.at[cindex,"fe_token_count_err_it1"] = avg_fe_err_token

                                        df_cost.at[cindex,"pp_error_count"] = avg_pp_error
                                        df_cost.at[cindex,"fe_error_count"] = avg_fe_error

                                        tmp_time = (df_ds['time_total']).mean()
                                        
                                        prompt_exe[config] = f"{tmp_time:.2f}"  
                                        if tmp_time <= 10:
                                            tmp_time=10 
                                        prompt_exe["CatDBChain_min"] = f"{tmp_time/60:.2f}" 
                                        prompt_exe["CatDBChain_10_min"] = f"{tmp_time/6:.2f}"   

                                    elif config == "CAAFE":
                                        df_cost.at[cindex,"config"]=f"{config}{cls}"
                                        df_cost.at[cindex,"token_count_it1"] = df_ds.loc[df_ds['number_iteration'] == 1, 'all_token_count'].values[0]

                                        max_iteration = df_ds['number_iteration'].max()  
                                        tmp_time = df_ds['time_total'].mean() 

                                        # missed iterations
                                        missed_count = 10 - len(df_ds['number_iteration'])
                                        missed_tokens = 0
                                        if missed_count > 0:
                                            base_prompt = df_ds.loc[df_ds['number_iteration'] == max_iteration, 'prompt_token_count'].values[0]
                                            missed_tokens = (missed_count +1)* base_prompt                                            
                                                    
                                        df_cost.at[cindex,"tokens_count"] = df_ds.loc[df_ds['number_iteration'] == max_iteration, 'all_token_count'].values[0] + missed_tokens
                                        prompt_exe[f"{config}{cls}"] = f"{tmp_time:.2f}"
                                        prompt_exe[f"{config}{cls}_min"] = f"{tmp_time/60:.2f}"
                                        prompt_exe[f"{config}{cls}_10_min"] = f"{tmp_time/6:.2f}"

                                    elif config == "AIDE":
                                        df_cost.at[cindex,"config"]="AIDE"                                                                             
                                        missed_count = 10 - len(df_ds)
                                        missed_tokens = 0
                                        base_prompt = df_ds.loc[df_ds['number_iteration'] == first_itr, 'all_token_count'].values[0]
                                        if missed_count > 0:                                            
                                            missed_tokens = (missed_count +1)* base_prompt                                            

                                        if last_itr -10 > 0:
                                            missed_tokens += base_prompt * (last_itr - 10)

                                        df_cost.at[cindex,"tokens_count"] = df_ds["all_token_count"].sum() + missed_tokens

                                        df_ds_tmp = df_ds.head(1).reset_index(drop=True)
                                        df_cost.at[cindex,"token_count_it1"] = df_ds_tmp.loc[0, 'all_token_count'] + base_prompt * (first_itr -1)
                                        df_cost.at[cindex,"token_count_err_it1"] = 0
                                        df_cost.at[cindex,"error_count"] = first_itr

                                        if last_itr < 10:
                                            last_itr = 10
                                        tmp_time = (df_ds['time_total'].mean() * last_itr) /10

                                        if tmp_time <= 40:
                                            tmp_time=40 
                                        prompt_exe[config] = f"{tmp_time:.2f}"  
                                        prompt_exe["AIDE_min"] = f"{tmp_time/60:.2f}"  
                                        prompt_exe["AIDE_10_min"] = f"{tmp_time/6:.2f}"  

                                    if config == "AutoGen":
                                        df_cost.at[cindex,"config"]="AutoGen"
                                        df_cost.at[cindex,"tokens_count"] = int(df_ds['prompt_token_count'].mean() * 10)  

                                        df_ds_tmp = df_ds.sort_values(by='all_token_count').reset_index(drop=True).head(1)

                                        df_cost.at[cindex,"token_count_it1"] = int(df_ds_tmp.loc[0, 'prompt_token_count']) + int((df_ds_tmp.loc[0,'all_token_count'] - df_ds_tmp.loc[0, 'prompt_token_count']) )
                                        df_cost.at[cindex,"token_count_err_it1"] = 0
                                        df_cost.at[cindex,"error_count"] = df_ds_tmp.loc[0,'number_iteration_error']

                                        tmp_time = df_ds['time_total'].mean() 
                                        if tmp_time <= 10:
                                            tmp_time=10                                        
                                        prompt_exe[config] = f"{tmp_time:.2f}" 
                                        prompt_exe["AutoGen_min"] = f"{tmp_time/60:.2f}"  
                                        prompt_exe["AutoGen_10_min"] = f"{tmp_time/6:.2f}"    
    
                                        
                    if des == "No" and (ds.endswith("rnc") or ((index in {106,107,108,109,1010,105} and llm == 'gemini-1.5-pro-latest') or index in {101,102,103,104})):    
                        if task_type in {"binary", "multiclass"}:
                            tsk = "classification"
                        else:
                            tsk = "regression"                  

                        dataset_load_time = df_csv_read.loc[df_csv_read['dataset']==ds]["time"].values[0] / 1000
                        df_exe.loc[len(df_exe)] = [ ds, ds_title,0,0,0, prompt_exe["CatDB"], prompt_exe["CatDBChain"], prompt_exe["CAAFETabPFN"],prompt_exe["CAAFERandomForest"], prompt_exe["AIDE"], prompt_exe["AutoGen"],
                            prompt_exe["CatDB_min"], prompt_exe["CatDBChain_min"], prompt_exe["CAAFETabPFN_min"],prompt_exe["CAAFERandomForest_min"], prompt_exe["AIDE_min"], prompt_exe["AutoGen_min"],
                                prompt_exe["CatDB_10_min"], prompt_exe["CatDBChain_10_min"], prompt_exe["CAAFETabPFN_10_min"],prompt_exe["CAAFERandomForest_10_min"],prompt_exe["AIDE_min"], prompt_exe["AutoGen_10_min"],
                                                    dataset_load_time, llm, des, task_type, tsk, samples]  

                        #df_automl_exe = pd.DataFrame(columns = ["dataset_name","dataset_name_orig", "time","dataset_load_time", "llm_model"])
                        ds_config_corr = dataset_corr[ds]
                        if ds_config_corr == "CatDB":
                            automl_time =   prompt_exe["CatDB"]
                        else:
                            automl_time =   prompt_exe["CatDBChain"]

                        df_automl_exe.loc[len(df_automl_exe)] = [ds, ds_title, automl_time, dataset_load_time, llm ]
    # Add Local Execution Time
    df_local = load_results(f"{root_path}/raw_results/Experiment1_Local_Pipeline.dat")
    df_AUG_runtime = load_results(f"{root_path}/Experiment1_Augmentation.dat")
    df_Cleaning_runtime = load_results(f"{root_path}/Experiment1_Cleaning_Time.dat")
    df_micro = pd.DataFrame(columns = df_cleaning_runtime.columns)
    

    for (ds,ds_title, task_type, index) in datasets:
        if index not in {106,107,108,109,1010,105}:
            continue
        corr_config = dataset_corr[ds]
        time_m = 0
        time_g = 0
        for run_mod in {'M', 'G'}:
            if task_type in {"binary", "multiclass"}:
                tsk = "classification"
            else:
                tsk = "regression"

            df_m = df_local.loc[(df_local['dataset_name'] == ds) & 
                                (df_local['config'] == corr_config) &
                                (df_local['llm_model'] == 'gemini-1.5-pro-latest') &
                                (df_local['status'] == True) &
                                (df_local['has_description'] == "No") &
                                (df_local['operation'] == f"Run-Local-Pipeline-{run_mod}")]
            
            df_saga = df_Cleaning_runtime.loc[(df_Cleaning_runtime['dataset'] == ds)]
            df_aug = df_AUG_runtime.loc[(df_AUG_runtime['dataset'] == ds)]
            tmp_time_saga = 0
            tmp_time_aug = 0
            if len(df_saga) > 0:
                tmp_time_saga = (df_saga['time']).mean() 

            if len(df_aug) > 0:
                tmp_time_aug = (df_aug['time']).mean()     
            
            tmp_time = (df_m['time_execution']).mean()
            time_min = tmp_time / 60 
            tmp_time = f"{tmp_time:.1f}"
            time_min = f"{time_min:.1f}" 
            if run_mod == "M":
                time_m = tmp_time
                time_m_min = time_min
            else:
                time_g = tmp_time  
                time_g_min = time_min  
        df_exe.loc[len(df_exe)] = [ds, f"{ds_title}-MG", time_m, time_g, tmp_time_saga, 0,0,0,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, 'gemini-1.5-pro-latest', 'No', task_type, tsk, 0]
        
        time_CAAFETabPFN = df_exe.loc[(df_exe['dataset_name'] == ds) & (df_exe['llm_model'] == 'gemini-1.5-pro-latest'), 'CAAFETabPFN'].values[0]
        time_CAAFERF = df_exe.loc[(df_exe['dataset_name'] == ds) & (df_exe['llm_model'] == 'gemini-1.5-pro-latest'), 'CAAFERandomForest'].values[0]
        time_AIDE = df_exe.loc[(df_exe['dataset_name'] == ds) & (df_exe['llm_model'] == 'gemini-1.5-pro-latest'), 'AIDE'].values[0]
        time_AutoGen = df_exe.loc[(df_exe['dataset_name'] == ds) & (df_exe['llm_model'] == 'gemini-1.5-pro-latest'), 'AutoGen'].values[0]
        df_cleaning_runtime.loc[len(df_cleaning_runtime)] = [ds_title,time_m, time_g, time_CAAFETabPFN, time_CAAFERF, time_AIDE, time_AutoGen, f"{tmp_time_saga/1000:.1f}"]

        cindex = len(df_micro)
        df_micro.loc[cindex] = [ds_title, f" &{time_m}", f" & {time_g}", f" & {time_CAAFETabPFN}", f" & {time_CAAFERF}", f" & {time_AIDE}", f" & {time_AutoGen}", f" & {tmp_time_saga/1000:.1f} + {tmp_time_aug/1000:.1f} \\\\ \\chline"]

    # find AVG, SUM, number of failed
    
    df_ten_runtime = pd.DataFrame(columns = ["config", "faile_gemini", "avg_gemini", "sum_gemini", "faile_llama", "avg_llama", "sum_llama", "faile_gpt-4o", "avg_gpt-4o", "sum_gpt-4o"])
    df_exe_tmp = df_exe[df_exe['dataset_name_orig'].isin(["Airline", "IMDB-IJS", "Accidents", "Financial", "CMC", "Bike-Sharing", "House-Sales", "NYC"])]

    for config in ["CatDB_min", "CatDBChain_min", "CAAFETabPFN_min", "CAAFERandomForest_min", "AIDE_min","AutoGen_min"]:
        str_row = [f"{config.replace('_min', '')}"]
        for llm in llms:
            df_runtime = df_exe_tmp.loc[(df_exe['llm_model'] == llm)]
            failed = len(df_runtime.loc[(df_runtime[config].astype(float) == 0.001)])
            if llm == 'gpt-4o':
                str_nline = '\\\ \chline'
            else:
                str_nline = ''

            df_runtime = df_runtime.loc[(df_runtime[config].astype(float) > 0.001)]
            avg = df_runtime[config].astype(float).mean()
            sum = df_runtime[config].astype(float).sum()
            str_row.append(f"& {failed}") 
            str_row.append(f"& {avg:0.1f}") 
            str_row.append(f"& {sum:0.1f} {str_nline}")
        #str_row[len(str_row)] = f"{str_row[len(str_row)]} \\ \chline"    
        df_ten_runtime.loc[len(df_ten_runtime)] = str_row    
        #           

    # Cost Log scale modify:
    for ds in ["CMC", "Tic-Tac-Toe"]:
        for col in ["token_count_it1","pp_token_count_it1","fe_token_count_it1", "token_count_err_it1","pp_token_count_err_it1","fe_token_count_err_it1"]:
            df_cost.loc[(df_cost['dataset_name_orig'] == ds) & (df_cost[col] == 0), col] = 0.001



    #
    df_sort.to_csv(f"{root_path}/AllResults.csv", index=False)
    df_cost.to_csv(f"{root_path}/CostResults.csv", index=False)
    df_exe.to_csv(f"{root_path}/ExeResults.csv", index=False)   
    df_automl_exe.to_csv(f"{root_path}/AutoMLExeResults.csv", index=False)  
    df_cleaning_runtime.to_csv(f"{root_path}/CleaningExeResults.csv", index=False)  

    # 
    fname = f"{root_path}/tbl_cleaning_runtime.txt"
    df_micro.to_csv(fname, index=False, header=True)
    replace_comma(fname=fname)

    fname = f"{root_path}/tbl_runtime.txt"
    df_ten_runtime.to_csv(fname, index=False, header=True)
    replace_comma(fname=fname)
 
