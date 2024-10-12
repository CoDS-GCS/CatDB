import pandas as pd
from MergeResults import load_merge_all_results, get_top_k_binary, get_top_k_multiclass, replace_comma, mergse_dfs, get_top_k_all, dataset_corr


if __name__ == '__main__':
    root_path = "../results"        
    df_pip_gen = load_merge_all_results(root_path=root_path)

    micor_tbl_cols = ["dataset_name", "Metric" , "llm_model",
                                       "CatDB_train_auc","CatDB_test_auc",
                                       "CatDBChain_train_auc","CatDBChain_test_auc",
                                       "CAAFETabPFN_train_auc","CAAFETabPFN_test_auc",
                                       "CAAFERandomForest_train_auc","CAAFERandomForest_test_auc",
                                       "AutoSklearn_train_auc","AutoSklearn_test_auc",
                                       "H2O_train_auc","H2O_test_auc",
                                       "Flaml_train_auc","Flaml_test_auc",
                                       "Autogluon_train_auc","Autogluon_test_auc"]
    df_micro = pd.DataFrame(columns = micor_tbl_cols)   

    datasetIDs = [("Airline","Airline",101),
                  ("IMDB-IJS","IMDB-IJS",102),
                  ("Accidents","Accidents",103),
                  ("Financial","Financial",104),                  
                  ("CMC","oml_dataset_3_rnc",3),                 
                  ("Tic-Tac-Toe","oml_dataset_6_rnc",6),                   
                  ("Walking-Activity","oml_dataset_20_rnc",13),
                  ("Bike-Sharing","oml_dataset_22_rnc",15),
                  ("House-Sales","oml_dataset_23_rnc",16),
                  ("NYC","oml_dataset_24_rnc",17)]
    
    
    datasets = []
    for ds in datasetIDs:
        ds_name, ds_rnc_name, index = ds
        datasets.append((ds_rnc_name, f"{ds_name}", index))

    llms = ["gpt-4o", "gemini-1.5-pro-latest","llama-3.1-70b-versatile"]

    wins = { "gemini-1.5-pro-latest":{ "CatDB_train_auc":0,"CatDB_test_auc":0,
                        "CatDBChain_train_auc":0,"CatDBChain_test_auc":0,
                        "CAAFETabPFN_train_auc":0,"CAAFETabPFN_test_auc":0,
                        "CAAFERandomForest_train_auc":0,"CAAFERandomForest_test_auc":0,
                        "AutoSklearn_train_auc":0,"AutoSklearn_test_auc":0,
                        "H2O_train_auc":0,"H2O_test_auc":0,
                        "Flaml_train_auc":0,"Flaml_test_auc":0,
                        "Autogluon_train_auc":0,"Autogluon_test_auc":0},
            "llama-3.1-70b-versatile":{"CatDB_train_auc":0,"CatDB_test_auc":0,
                        "CatDBChain_train_auc":0,"CatDBChain_test_auc":0,
                        "CAAFETabPFN_train_auc":0,"CAAFETabPFN_test_auc":0,
                        "CAAFERandomForest_train_auc":0,"CAAFERandomForest_test_auc":0,
                        "AutoSklearn_train_auc":0,"AutoSklearn_test_auc":0,
                        "H2O_train_auc":0,"H2O_test_auc":0,
                        "Flaml_train_auc":0,"Flaml_test_auc":0,
                        "Autogluon_train_auc":0,"Autogluon_test_auc":0},
            "gpt-4o":{"CatDB_train_auc":0,"CatDB_test_auc":0,
                        "CatDBChain_train_auc":0,"CatDBChain_test_auc":0,
                        "CAAFETabPFN_train_auc":0,"CAAFETabPFN_test_auc":0,
                        "CAAFERandomForest_train_auc":0,"CAAFERandomForest_test_auc":0,
                        "AutoSklearn_train_auc":0,"AutoSklearn_test_auc":0,
                        "H2O_train_auc":0,"H2O_test_auc":0,
                        "Flaml_train_auc":0,"Flaml_test_auc":0,
                        "Autogluon_train_auc":0,"Autogluon_test_auc":0},            
                        }
    
    wins2 = dict()
    for llm in llms:
        llm_key = dict()
        for key in wins[llm].keys():
            llm_key[key] = 0
        wins2[llm] = llm_key    
    
    classifier = ["Auto", "TabPFN", "RandomForest"]

    llms_shorname = {"gemini-1.5-pro-latest":"Gemini-1.5","llama-3.1-70b-versatile":"Llama3.1-70b", "gpt-4o": "GPT-4o"}

    configs = ["CatDB", "CatDBChain", "CAAFE","AutoSklearn","H2O","Flaml","Autogluon"]    
    ds_id = None
    for (ds,ds_title, index) in datasets:
        for llm in llms:
            tbl_data = {"CatDB_train_auc":0,"CatDB_test_auc":0,
                        "CatDBChain_train_auc":0,"CatDBChain_test_auc":0,
                        "CAAFETabPFN_train_auc":0,"CAAFETabPFN_test_auc":0,
                        "CAAFERandomForest_train_auc":0,"CAAFERandomForest_test_auc":0,
                        "AutoSklearn_train_auc":0,"AutoSklearn_test_auc":0,
                        "H2O_train_auc":0,"H2O_test_auc":0,
                        "Flaml_train_auc":0,"Flaml_test_auc":0,
                        "Autogluon_train_auc":0,"Autogluon_test_auc":0}
            for des in {"Yes", "No"}:                
                for config in configs:
                    for cls in classifier:                    
                            df = df_pip_gen.loc[(df_pip_gen['dataset_name'] == ds) & 
                                                (df_pip_gen['config'] == config) &
                                                (df_pip_gen['llm_model'] == llm) &
                                                (df_pip_gen['status'] == True) &
                                                #(df_pip_gen['classifier'] == cls) &
                                                (df_pip_gen['has_description'] == des) &
                                                ((df_pip_gen['operation'] == 'Run-Pipeline') | 
                                                 (df_pip_gen['operation']=='Run-CAAFE') |
                                                 (df_pip_gen['operation']=='Run-AutoML'))]
                            
                            
                            if config not in {"AutoSklearn","H2O","Flaml","Autogluon"}:
                                df = df.loc[df['classifier'] == cls]
                            else:
                               df = df.loc[df['number_iteration'] == 1].head(1)     

                            if len(df) > 0:   
                                df_sort = pd.DataFrame(columns = df_pip_gen.columns)                                                        
                                
                                df_all_task = get_top_k_all(config=config, df=df, k=1)
                                df_all_task["number_iteration"] = [ki for ki in range(1, len(df_all_task)+1)]
                                mergse_dfs(df_sort, df_all_task) 

                                df_ds = df_sort.loc[(df_sort['dataset_name'] ==ds) &
                                                    (df_sort['llm_model'] == llm)&
                                                    (df_sort['config'] == config)] 
                                
                                if config not in {"AutoSklearn","H2O","Flaml","Autogluon"}:
                                    df_ds = df_ds.loc[df_ds['classifier'] == cls] 

                                if len(df_ds) == 0:
                                    continue

                                config_title = config
                                if config in {"CatDB", "CatDBChain"}:
                                    config_title = config
                                elif config == "CAAFE":
                                    if cls == "TabPFN":
                                        config_title="CAAFETabPFN"
                                    elif cls == "RandomForest":
                                        config_title = "CAAFERandomForest" 
                               
                               
                                task_type = df_ds.iloc[0]['task_type']
                                col_train = "train_auc"
                                col_test = "test_auc"              

                                key_train = f"{config_title}_train_auc"
                                key_test =  f"{config_title}_test_auc"

                                tbl_data[key_train] = int(df_ds.iloc[0][col_train] * 10000)
                                tbl_data[key_test] = int(df_ds.iloc[0][col_test] * 10000)
                                

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

            if llm == "gemini-1.5-pro-latest":
                if ds in {"IMDB-IJS", "oml_dataset_6_rnc"}:
                    df_micro.at[cindex,"Metric"] = f" & AUC"
                elif ds in {"Airline", "Accidents", "Financial", "oml_dataset_3_rnc", "oml_dataset_20_rnc"}:
                    df_micro.at[cindex,"Metric"] = f" & AUC-ovr "    
                elif ds in {"oml_dataset_22_rnc", "oml_dataset_23_rnc", "oml_dataset_24_rnc"}:
                    df_micro.at[cindex,"Metric"] = f" & $R^2$ "        
            else:
                 df_micro.at[cindex,"Metric"] = "&"

            tbl_line = "" 
            if llm == "gpt-4o":
             df_micro.at[cindex,"dataset_name"] = "\multirow{3}{*}{"+ds_title+"}"         
            else:
                df_micro.at[cindex,"dataset_name"] = "" 

            if llm == "llama-3.1-70b-versatile":    
                tbl_line = "\\chline"
            df_micro.at[cindex,"llm_model"] = "& "+llms_shorname[llm]
            for k in tbl_data.keys():
                htext = ""
                corr_config = dataset_corr[ds]
                line_text = ""
                if k == "Autogluon_test_auc":
                    line_text = f"\\\\ {tbl_line}"
               
                if (corr_config in {"CatDB"} and k in {"CatDB_train_auc", "CatDB_test_auc"}) or (corr_config in {"CatDBChain"} and k in {"CatDBChain_train_auc", "CatDBChain_test_auc"}):
                        htext="\\cellcolor{lightgray!50}"        

                if tbl_data[k] is None:
                    df_micro.at[cindex,k] = "& N/A" +  line_text 
                elif "test" in k and tbl_data[k] >= max_test:
                    df_micro.at[cindex,k] = "&"+htext+"\\textbf{"+f"{tbl_data[k]/100}"+"}" + line_text
                elif "train" in k and tbl_data[k] >= max_train:
                    df_micro.at[cindex,k] = "& "+htext+" \\textbf{"+f"{tbl_data[k]/100}"+"}" + line_text
                     
                elif tbl_data[k] == 0:                  
                    df_micro.at[cindex,k] = "&--" + line_text
                else:
                    df_micro.at[cindex,k] = f"& {htext} {tbl_data[k]/100}" + line_text

                

    fname = f"{root_path}/tbl_microbenchmark.csv"
    df_micro.to_csv(fname, index=False, header=None)
    replace_comma(fname=fname)