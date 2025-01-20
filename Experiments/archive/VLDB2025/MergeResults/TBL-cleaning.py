import pandas as pd
from MergeResults import load_merge_all_results, get_top_k_all, replace_comma, mergse_dfs, load_results, dataset_corr, dataset_corr_clean
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    root_path = "../results"        
    df_pip_gen = load_merge_all_results(root_path=root_path)
    
    df_local = load_results(f"{root_path}/raw_results/Experiment1_Local_Pipeline.dat")

    
    #"Metric" ,"Selected_Method",
    micor_tbl_cols = ["dataset_name", 
                      "CatDBM_train_auc","CatDBM_test_auc",     
                                       "CatDB_train_auc","CatDB_test_auc",
                                       "CAAFETabPFN_train_auc","CAAFETabPFN_test_auc",
                                       "CAAFERandomForest_train_auc","CAAFERandomForest_test_auc",
                                       "AIDE_train_auc","AIDE_test_auc",
                                       "AutoGen_train_auc","AutoGen_test_auc",
                                       "H2O_train_auc","H2O_test_auc",
                                       "Flaml_train_auc","Flaml_test_auc",
                                       "Autogluon_train_auc","Autogluon_test_auc","Cleaning",
                                       "H2O_Cleaning_train_auc","H2O_Cleaning_test_auc",
                                       "Flaml_Cleaning_train_auc","Flaml_Cleaning_test_auc",
                                       "Autogluon_Cleaning_train_auc","Autogluon_Cleaning_test_auc"]
    df_micro = pd.DataFrame(columns = micor_tbl_cols)   
    datasetIDs = [("EU-IT","EU-IT",106),  
                  ("WiFi","WiFi",109), 
                  ("Etailing","Etailing",107),                 
                  ("Survey","Midwest-Survey",108),                                   
                  ("Utility","Utility",1010),
                  ("Yelp","Yelp",105),
                ]
    
    df_overall = pd.DataFrame(columns = ["dataset_name","llm_model","CatDB", "CatDBChain","max_other"])
    
    datasets = []
    for ds in datasetIDs:
        ds_name, ds_rnc_name, index = ds
        datasets.append((ds_rnc_name, f"{ds_name}", index))

    llms = ["gemini-1.5-pro-latest"]

    wins = { "gemini-1.5-pro-latest":{ "CatDB_train_auc":0,"CatDB_test_auc":0,
                        "CatDBChain_train_auc":0,"CatDBChain_test_auc":0,
                        "CAAFETabPFN_train_auc":0,"CAAFETabPFN_test_auc":0,
                        "CAAFERandomForest_train_auc":0,"CAAFERandomForest_test_auc":0,
                        "AIDE_train_auc":0,"AIDE_test_auc":0,
                        "AutoGen_train_auc":0,"AutoGen_test_auc":0,
                        "H2O_train_auc":0,"H2O_test_auc":0,
                        "Flaml_train_auc":0,"Flaml_test_auc":0,
                        "Autogluon_train_auc":0,"Autogluon_test_auc":0,
                        "H2O_Cleaning_train_auc":0,"H2O_Cleaning_test_auc":0,
                        "Flaml_Cleaning_train_auc":0,"Flaml_Cleaning_test_auc":0,
                        "Autogluon_Cleaning_train_auc":0,"Autogluon_Cleaning_test_auc":0},                     
                        }
    
    wins2 = dict()
    for llm in llms:
        llm_key = dict()
        for key in wins[llm].keys():
            llm_key[key] = 0
        wins2[llm] = llm_key    
    
    classifier = ["Auto", "TabPFN", "RandomForest"]
    sub_tasks = ["","SAGA", "Learn2Clean"]
    llms_shorname = {"gemini-1.5-pro-latest":"Gemini-1.5"}

    configs = ["CatDB", "CatDBChain", "CAAFE", "AIDE", "AutoGen","H2O","Flaml","Autogluon"]    
    ds_id = None
    for (ds,ds_title, index) in datasets:
        for llm in llms:
            tbl_data = {"CatDB_train_auc":0,"CatDB_test_auc":0,
                        "CatDBM_train_auc":0,"CatDBM_test_auc":0,
                        "CAAFETabPFN_train_auc":0,"CAAFETabPFN_test_auc":0,
                        "CAAFERandomForest_train_auc":0,"CAAFERandomForest_test_auc":0,
                        "AIDE_train_auc":0,"AIDE_test_auc":0,
                        "AutoGen_train_auc":0,"AutoGen_test_auc":0,
                        "H2O_train_auc":0,"H2O_test_auc":0,
                        "Flaml_train_auc":0,"Flaml_test_auc":0,
                        "Autogluon_train_auc":0,"Autogluon_test_auc":0,
                        "H2O_Cleaning_train_auc":0,"H2O_Cleaning_test_auc":0,
                        "Flaml_Cleaning_train_auc":0,"Flaml_Cleaning_test_auc":0,
                        "Autogluon_Cleaning_train_auc":0,"Autogluon_Cleaning_test_auc":0}
            for des in {"Yes", "No"}:                
                for config in configs:
                    for cls in classifier:
                        for st in sub_tasks:       
                            corr_config = dataset_corr[ds]  
                            corr_clean = dataset_corr_clean[ds]                
                            df = df_pip_gen.loc[(df_pip_gen['dataset_name'] == ds) & 
                                                (df_pip_gen['config'] == config) &
                                                (df_pip_gen['llm_model'] == llm) &
                                                (df_pip_gen['status'] == True) &
                                                #(df_pip_gen['classifier'] == cls) &
                                                (df_pip_gen['has_description'] == des) &
                                                ((df_pip_gen['operation'] == 'Run-Pipeline') | 
                                                (df_pip_gen['operation']=='Run-CAAFE') |
                                                (df_pip_gen['operation']=='Run-AutoML'))]                   
                                                       
                            if config in {"AutoSklearn","H2O","Flaml","Autogluon"} and st in {"SAGA", "Learn2Clean"}: 
                                df = df.loc[df['sub_task'] == corr_clean]
                                if len(df) == 0:
                                    continue

                            if config not in {"AutoSklearn","H2O","Flaml","Autogluon"} and st in {"SAGA", "Learn2Clean"}:
                                continue   

                            if config not in {"AutoSklearn","H2O","Flaml","Autogluon"}:
                                df = df.loc[df['classifier'] == cls]
                            else:
                               df = df.loc[df['number_iteration'] == 1].head(1)     

                            if len(df) > 0:   
                                df_sort = pd.DataFrame(columns = df_pip_gen.columns)                                                        
                                
                                df_all_task = get_top_k_all(config=config, df=df, k=1)
                                df_all_task["number_iteration"] = [ki for ki in range(1, len(df_all_task)+1)]
                                mergse_dfs(df_sort, df_all_task) 

                                if len(df_sort) == 0:
                                    continue                              

                                df_ds = df_sort.loc[(df_sort['dataset_name'] ==ds) &
                                                    (df_sort['llm_model'] == llm)&
                                                    (df_sort['config'] == config)] 
                                
                                if config not in {"AutoSklearn","H2O","Flaml","Autogluon"}:
                                    df_ds = df_ds.loc[df_ds['classifier'] == cls] 


                                config_title = config
                                if config in {"CatDB", "CatDBChain"}:                                    
                                    if corr_config != config:
                                        continue
                                    else:
                                        config_title = "CatDB"
                                elif config == "CAAFE":
                                    if cls == "TabPFN":
                                        config_title="CAAFETabPFN"
                                    elif cls == "RandomForest":
                                        config_title = "CAAFERandomForest" 
                                elif config in {"AIDE", "AutoGen"}:
                                    config_title = config

                                elif config in {"AutoSklearn","H2O","Flaml","Autogluon"}:
                                    if st in {"SAGA", "Learn2Clean"}:
                                        config_title = f"{config}_Cleaning"


                                task_type = df_ds.iloc[0]['task_type']
                                col_train = "train_auc"
                                col_test = "test_auc"              

                                key_train = f"{config_title}_train_auc"
                                key_test =  f"{config_title}_test_auc"

                                tbl_data[key_train] = int(df_ds.iloc[0][col_train] * 1000)
                                tbl_data[key_test] = int(df_ds.iloc[0][col_test] * 1000)

                                
                               
            corr_config = dataset_corr[ds]                    
            df_m = df_local.loc[(df_local['dataset_name'] == ds) & 
                                (df_local['config'] == corr_config) &
                                (df_local['llm_model'] == llm) &
                                (df_local['status'] == True) &
                                (df_local['has_description'] == "No") &
                                (df_local['operation'] == 'Run-Local-Pipeline-M')]
            df_m =   get_top_k_all(config=corr_config, df=df_m, k=1)  

            if len(df_m) > 0:
                tbl_data["CatDBM_train_auc"] = int(df_m.iloc[0]["train_auc"] * 1000)
                tbl_data["CatDBM_test_auc"] = int(df_m.iloc[0]["test_auc"] * 1000)


            max_train = 0
            max_test = 0
            max_other = 0
            for k in tbl_data.keys():
                if "test" in k:
                    max_test = max(max_test, tbl_data[k])
                    if "CatDB" not in k:
                        max_other = max(max_other, tbl_data[k])
                elif "train" in k:
                    max_train = max(max_train, tbl_data[k]) 

            
            cindex = len(df_micro)
            df_micro.loc[cindex] = [None for ti in micor_tbl_cols]            
           
            sch = "\\ding{86}"
            if corr_config == "CatDBChain":
                sch = "\\ding{59}"
    
            df_micro.at[cindex,"dataset_name"] =ds_title+"$^{\\text{"+sch+"}}$" 
            if corr_clean == "Learn2Clean":
                corr_clean = "L2C"
            df_micro.at[cindex,"Cleaning"] =f"& {corr_clean}" 
             
            #df_micro.at[cindex,"Selected_Method"] = f" & {corr_config} "  

            # if ds in {"WiFi"}:
            #     df_micro.at[cindex,"Metric"] = f" & AUC "
            # elif ds in {"EU-IT"}:
            #     df_micro.at[cindex,"Metric"] = f" & ACC "    
            # elif ds in {"Utility"}:
            #     df_micro.at[cindex,"Metric"] = f" & $R^2$ "
            # else:
            #    df_micro.at[cindex,"Metric"] = f" & AUC-ovr "           
                          
            
            tbl_line = " \\\\ \\chline"
            for k in tbl_data.keys(): 
                tl = ''        
                if k == 'Autogluon_Cleaning_test_auc':
                    tl = tbl_line

                if tbl_data[k] is None:
                    df_micro.at[cindex,k] = "& N/A"   
                elif "test" in k and tbl_data[k] >= max_test:
                    df_micro.at[cindex,k] = "& \\textbf{"+f"{tbl_data[k]/10}"+"}"+tl
                elif "train" in k and tbl_data[k] >= max_train:
                    df_micro.at[cindex,k] = "& \\textbf{"+f"{tbl_data[k]/10}"+"}" +tl 
                elif tbl_data[k] == 0:
                    df_micro.at[cindex,k] = "& -- "+tl    
                else:
                    df_micro.at[cindex,k] = f"& {tbl_data[k]/10}"+tl

    fname = f"{root_path}/tbl_cleaning.txt"
    df_micro.to_csv(fname, index=False, header=True)
    replace_comma(fname=fname)
