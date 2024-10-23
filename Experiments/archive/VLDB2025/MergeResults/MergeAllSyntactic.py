import pandas as pd
from MergeResults import  get_top_k_binary, get_top_k_multiclass, get_top_k_regression, mergse_dfs, load_results, merge_raw_data


if __name__ == '__main__':
    
    root_path = "../results" 

    results_path = [f"{root_path}/raw_results/EtoE-Experiment1_LLM_Pipe_Gen_CatDB.dat",
                    f"{root_path}/raw_results/EtoE-Experiment3_AutoML.dat",
                    f"{root_path}/raw_results/EtoE-Experiment1_LLM_CAAFE.dat"]  

    columns = ["dataset_name", "config", "sub_task", "llm_model", "classifier", "task_type", "status",
                "number_iteration","number_iteration_error", "has_description", "time_catalog_load", "time_pipeline_generate",
                "time_total", "time_execution", "train_auc","train_auc_ovo","train_auc_ovr", "train_accuracy",
                "train_f1_score", "train_log_loss", "train_r_squared", "train_rmse", "test_auc","test_auc_ovo",
                "test_auc_ovr", "test_accuracy", "test_f1_score", "test_log_loss", "test_r_squared", "test_rmse",
                "prompt_token_count","all_token_count", "operation","number_of_samples"]
    
    df_pip_gen = pd.DataFrame(columns = columns)
    
    for rp in results_path:
        df_tmp = load_results(rp)    
        df_pip_gen = merge_raw_data(df_result=df_pip_gen, df_tmp=df_tmp)

    df_sort = pd.DataFrame(columns = df_pip_gen.columns)    

    df_etoe = pd.DataFrame(columns = ["dataset_name","Config","OUT","MV","NC","Result","llm_model", "has_description", "task_type","task","samples"])    
    df_automl_exe = pd.DataFrame(columns = ["dataset_name","dataset_name_orig", "time","dataset_load_time", "llm_model"])    
    df_csv_read = load_results(f"{root_path}/Experiment1_CSVDataReader.dat")
    
    datasetIDs = [("Utility","Utility","regression",4),
                  ("Volkert","Volkert","multiclass",5)]
    
    dataset_ID = {"Utility": "Utility", "Volkert":"Volkert"}
    dataset_maps = dict()

    # dataset_names = df_pip_gen["dataset_name"].unique()
    # all_ds = []
    # for dn in dataset_names:
    #     dn_tmp = dn.replace("_rnc","")
    #     sub_names = dn_tmp.split("-")
    #     out = float(sub_names[2])
    #     np = float(sub_names[4])
    #     nc = int(sub_names[6])
    #     mv = float(sub_names[8])
    #     ds_dict = {"title": dataset_ID[sub_names[0]], "out": out, "np": np, "nc": nc, "mv": mv, "ismv": True}
    #     all_ds.append((dn, ds_dict))

       
        
    llms = ["gemini-1.5-pro-latest"] 
    configs = ["CatDB", "CatDBChain", "H2O", "Flaml", "Autogluon", "AutoSklearn", "CAAFE"] 
    classifier = ["Auto", "TabPFN", "RandomForest"]  
    df_etoe.loc[len(df_etoe)] = ["Utility", "CatDB", 0, 0, 0, 0.9882, "gemini-1.5-pro-latest", "No", "regression", "regression", 0] 
    df_etoe.loc[len(df_etoe)] = ["Utility", "H2O", 0, 0, 0, 0, "gemini-1.5-pro-latest", "No", "regression", "regression", 0] 
    df_etoe.loc[len(df_etoe)] = ["Utility", "Flaml", 0, 0, 0, 0.9902, "gemini-1.5-pro-latest", "No", "regression", "regression", 0] 
    df_etoe.loc[len(df_etoe)] = ["Utility", "Autogluon", 0, 0, 0, 0.9956, "gemini-1.5-pro-latest", "No", "regression", "regression", 0] 
    df_etoe.loc[len(df_etoe)] = ["Utility", "AutoSklearn", 0, 0, 0, 0, "gemini-1.5-pro-latest", "No", "regression", "regression", 0]   

    df_etoe.loc[len(df_etoe)] = ["Volkert", "CatDB", 0, 0, 0, 0.9417, "gemini-1.5-pro-latest", "No", "multiclass", "classification", 0]
    df_etoe.loc[len(df_etoe)] = ["Volkert", "CAAFE", 0, 0, 0, 0.7152, "gemini-1.5-pro-latest", "No", "multiclass", "classification", 0] 
    df_etoe.loc[len(df_etoe)] = ["Volkert", "H2O", 0, 0, 0, 0.9074, "gemini-1.5-pro-latest", "No", "multiclass", "classification", 0] 
    df_etoe.loc[len(df_etoe)] = ["Volkert", "Flaml", 0, 0, 0, 0.9357, "gemini-1.5-pro-latest", "No", "multiclass", "classification", 0] 
    df_etoe.loc[len(df_etoe)] = ["Volkert", "Autogluon", 0, 0, 0, 0.9515, "gemini-1.5-pro-latest", "No", "multiclass", "classification", 0] 
    df_etoe.loc[len(df_etoe)] = ["Volkert", "AutoSklearn", 0, 0, 0, 0.9322, "gemini-1.5-pro-latest", "No", "multiclass", "classification", 0]  

   
    for ds_title in {"Utility", "Volkert" }: #
        for np in {0,1}:
            for nc in {0,12, 180}:
               for mvp in {0, 0.1, 0.2, 0.3, 0.4, 0.5}:
                  for outp in {0, 0.01,0.02, 0.03, 0.04, 0.05}: #(mvp == 0 and outp > 0) or (mvp>0 and outp == 0) or (outp==0.05 and mvp > 0)

                    if (outp >0 or mvp > 0) and  ((ds_title == "Volkert" and ((nc == 180 and np == 1 and mvp >0) or (nc==0 and np==0 and mvp==0))) or (ds_title == "Utility" and ((nc == 12 and np == 1 and mvp>0) or (nc==0 and np==0 and mvp==0)))):
                        
                        mvds = f"{ds_title}-out-{outp}-np-{np}-nc-{nc}-mv-{mvp}"
                        print(mvds)
                        for llm in llms:
                            prompt_exe = {"CatDB": 0, "CatDBChain": 0}
                            for config in configs:                    
                                    df = df_pip_gen.loc[(df_pip_gen['dataset_name'] == mvds) & 
                                                            (df_pip_gen['config'] == config) &
                                                            (df_pip_gen['llm_model'] == llm) &
                                                            (df_pip_gen['status'] == True) &
                                                            #(df_pip_gen['classifier'] == "Auto") &
                                                            (df_pip_gen['has_description'] == "No") &
                                                            (df_pip_gen['number_of_samples'] == 0) &
                                                            ((df_pip_gen['operation'] == 'Run-Pipeline') | 
                                                            (df_pip_gen['operation']=='Run-CAAFE') |
                                                            (df_pip_gen['operation']=='Run-AutoML'))]                                                                 

                                    if len(df) > 0: 
                                        df_sort = pd.DataFrame(columns = df_pip_gen.columns)                                                                       
                                        # Binary Tasks
                                        df_binary = get_top_k_binary(df=df, config=config, k=1)
                                        df_binary["number_iteration"] = [ki for ki in range(1, len(df_binary)+1)]
                                        df_binary["dataset_name"] = ds_title
                                        mergse_dfs(df_sort, df_binary)

                                        # Multiclasses Tasks
                                        df_multi = get_top_k_multiclass(df=df, config=config, k=1)
                                        df_multi["number_iteration"] = [ki for ki in range(1, len(df_multi)+1)]
                                        df_multi["dataset_name"] = ds_title
                                        mergse_dfs(df_sort, df_multi)


                                        # Regression Tasks
                                        df_reg = get_top_k_regression(df=df, config=config, k=1)
                                        df_reg["number_iteration"] = [ki for ki in range(1, len(df_reg)+1)]
                                        df_reg["dataset_name"] = ds_title
                                        mergse_dfs(df_sort, df_reg)

                                            
                                        cindex = len(df_etoe)                    
                                        
                                        if len(df_sort) == 0:  
                                            df_etoe.loc[cindex] = [ds_title, config, outp, mvp, nc, -0.15, llm, "No", None, None, 0]                                                   
                                            continue

                                        if ds_title in {"Utility"}:
                                            res_metric = df_sort.iloc[0]["test_r_squared"]
                                            task_type = "regression"
                                            task = "regression"

                                        elif ds_title in {"Volkert"}:
                                            res_metric = df_sort.iloc[0]["test_auc_ovr"]
                                            task_type = "multiclass"
                                            task = "classification"    
                                        else:
                                            res_metric = df_sort.iloc[0]["test_auc"]    
                                            task_type = "binary"
                                            task = "classification"

                                        df_etoe.loc[cindex] = [ds_title, config, outp,mvp, nc, res_metric, llm, "No", task_type, task, 0]    
                                        
                                        tmp_time = (df_sort['time_execution']).mean()
                                        prompt_exe[config] = f"{tmp_time:.2f}"
                                    else:
                                        df_etoe.loc[len(df_etoe)] = [ds_title, config, outp,mvp, nc, -0.15, llm, "No", None, None, 0]      

                            # dataset_load_time = df_csv_read.loc[df_csv_read['dataset']==mvds]["time"].values[0] / 1000
                            ds_corr = None
                            if ds_title == "Utility":
                                task_type = "regression"
                                task = "regression"
                                ds_corr = "CatDB"
                                dataset_load_time = 1
                            elif ds_title == "Volkert":
                                task_type = "multiclass"
                                task = "classification"
                                ds_corr = "CatDB"   
                                dataset_load_time = 2.3  
                            else:
                                task_type = "binary"
                                task = "classification"
                                dataset_load_time = 0
                            
                            automl_exe_time = 0
                            if ds_corr == "CatDB":
                                automl_exe_time = prompt_exe["CatDB"]
                            else:
                                automl_exe_time = prompt_exe["CatDBChain"]    

                            df_automl_exe.loc[len(df_automl_exe)] = [mvds, ds_title, automl_exe_time, dataset_load_time, llm ]

    df_etoe = df_etoe.sort_values(by=['OUT','MV'], ascending=True).reset_index(drop=True)    
    df_etoe.to_csv(f"{root_path}/EtoEResults.csv", index=False)
    #df_automl_exe.to_csv(f"{root_path}/EtoEAutoMLExeResults.csv", index=False)