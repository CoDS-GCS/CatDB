import pandas as pd
from MergeResults import load_merge_all_results, get_top_k_binary, get_top_k_multiclass, replace_comma, mergse_dfs


if __name__ == '__main__':
    root_path = "/home/saeed/Documents/Github/CatDB/Experiments/archive/SIGMOD2025-Results/"        
    df_pip_gen = load_merge_all_results(root_path=root_path)

    micor_tbl_cols = ["dataset_name","llm_model",
                                       "CatDB_train_auc","CatDB_test_auc",
                                       "CatDBChain_train_auc","CatDBChain_test_auc",
                                       "CAAFETabPFN_train_auc","CAAFETabPFN_test_auc",
                                       "CAAFERandomForest_train_auc","CAAFERandomForest_test_auc",
                                       "AutoSklearn_train_auc","AutoSklearn_test_auc",
                                       "H2O_train_auc","H2O_test_auc",
                                       "Flaml_train_auc","Flaml_test_auc",
                                       "Autogluon_train_auc","Autogluon_test_auc",
                                       "CatDB_test_auc_diff","CatDBChain_test_auc_diff"]
    df_micro = pd.DataFrame(columns = micor_tbl_cols)   
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
                  ("Walking-Activity","oml_dataset_20_rnc",13)
                ]
    datasets = []
    for ds in datasetIDs:
        ds_name, ds_rnc_name, index = ds
        datasets.append((ds_rnc_name, f"{ds_name}", index))

    llms = ["gpt-4o", "gemini-1.5-pro-latest","llama3-70b-8192"]

    wins = { "gemini-1.5-pro-latest":{ "CatDB_train_auc":0,"CatDB_test_auc":0,
                        "CatDBChain_train_auc":0,"CatDBChain_test_auc":0,
                        "CAAFETabPFN_train_auc":0,"CAAFETabPFN_test_auc":0,
                        "CAAFERandomForest_train_auc":0,"CAAFERandomForest_test_auc":0,
                        "AutoSklearn_train_auc":0,"AutoSklearn_test_auc":0,
                        "H2O_train_auc":0,"H2O_test_auc":0,
                        "Flaml_train_auc":0,"Flaml_test_auc":0,
                        "Autogluon_train_auc":0,"Autogluon_test_auc":0},
            "llama3-70b-8192":{"CatDB_train_auc":0,"CatDB_test_auc":0,
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

    llms_shorname = {"gemini-1.5-pro-latest":"Gemini-1.5","llama3-70b-8192":"Llama3-70b", "gpt-4o": "GPT-4o"}

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
                                                ((df_pip_gen['sub_task'] == 'ModelSelection') | (pd.isnull(df_pip_gen['sub_task'])))]
                            
                            if config not in {"AutoSklearn","H2O","Flaml","Autogluon"}:
                                df = df.loc[df['classifier'] == cls]
                            else:
                               df = df.loc[df['number_iteration'] == 1].head(1)     

                            if len(df) > 0:   
                                df_sort = pd.DataFrame(columns = df_pip_gen.columns)                                                        
                                
                                df_binary = get_top_k_binary(config=config, df=df, k=1)
                                df_binary["number_iteration"] = [ki for ki in range(1, len(df_binary)+1)]
                                mergse_dfs(df_sort, df_binary)

                                # Multitask Tasks
                                df_multi = get_top_k_multiclass(df=df, config=config, k=1)
                                df_multi["number_iteration"] = [ki for ki in range(1, len(df_multi)+1)]
                                mergse_dfs(df_sort, df_multi)

                                df_ds = df_sort.loc[(df_sort['dataset_name'] ==ds) &
                                                    (df_sort['llm_model'] == llm)&
                                                    (df_sort['config'] == config)] 
                                
                                if config not in {"AutoSklearn","H2O","Flaml","Autogluon"}:
                                    df_ds = df_ds.loc[df_ds['classifier'] == cls] 


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
                                if task_type == "multiclass":
                                    col_train = f"{col_train}_ovr"
                                    col_test = f"{col_test}_ovr"
                                
                                if task_type == "regression":
                                    continue

                                key_train = f"{config_title}_train_auc"
                                key_test =  f"{config_title}_test_auc"

                                tbl_data[key_train] = int(df_ds.iloc[0][col_train] * 100000)
                                tbl_data[key_test] = int(df_ds.iloc[0][col_test] * 100000)
                                

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
            
            tbl_line = "" #"\\cmidrule{2-20}"
            if llm == "gpt-4o":
             df_micro.at[cindex,"dataset_name"] = "\multirow{3}{*}{"+ds_title+"}"         
            else:
                df_micro.at[cindex,"dataset_name"] = "" 

            if llm == "llama3-70b-8192":    
                tbl_line = "\\chline"
            df_micro.at[cindex,"llm_model"] = "& "+llms_shorname[llm]
            for k in tbl_data.keys():
                if tbl_data[k] is None:
                    df_micro.at[cindex,k] = "& N/A"   
                elif "test" in k and tbl_data[k] >= max_test:
                    df_micro.at[cindex,k] = "& \\textbf{"+f"{tbl_data[k]/1000}"+"}"
                    wins[llm][k] +=1
                elif "train" in k and tbl_data[k] >= max_train:
                    df_micro.at[cindex,k] = "& \\textbf{"+f"{tbl_data[k]/1000}"+"}"
                    wins[llm][k] +=1  
                elif tbl_data[k] == 0:
                     df_micro.at[cindex,k] = "& --" 
                else:
                    df_micro.at[cindex,k] = f"& {tbl_data[k]/1000}"

                if tbl_data[k] is  not None:   
                 if "test" in k and tbl_data[k]+4 >= max_test:
                    wins2[llm][k] +=1
                 elif "train" in k and tbl_data[k]+4 >= max_train:
                    wins2[llm][k] +=1      
            
            if tbl_data["CatDB_test_auc"] > 0: 
                catdb_value =(tbl_data["CatDB_test_auc"]-max_other)/1000
                
                if catdb_value < 0:
                    catdb_value_str = f"${catdb_value}$"
                elif catdb_value> 0:
                    catdb_value_str = f"$+{catdb_value}$"
                else:
                    catdb_value_str = "0.0"     
            else:
                 catdb_value_str = "--"


            if tbl_data["CatDBChain_test_auc"] > 0:     
                catdb_chain_value = (tbl_data["CatDBChain_test_auc"]-max_other) / 1000            

                if catdb_chain_value < 0:
                    catdb_chain_value_str = f"${catdb_chain_value}$"
                elif catdb_chain_value > 0:
                    catdb_chain_value_str = f"$+{catdb_chain_value}$"
                else:                 
                    catdb_chain_value_str = "0.0"        
            else:
                 catdb_chain_value_str = "--"       

            df_micro.at[cindex,"CatDB_test_auc_diff"] = f'& {catdb_value_str}'
            df_micro.at[cindex,"CatDBChain_test_auc_diff"] = f'& {catdb_chain_value_str} \\\\ {tbl_line}'

            
    # add leader board:
    for llm in llms:
        cindex = len(df_micro)
        df_micro.loc[cindex] = [None for ti in micor_tbl_cols]

        if llm == "gpt-4o":
             df_micro.at[cindex,"dataset_name"] = "Leader"      
        elif llm == "gemini-1.5-pro-latest":
            df_micro.at[cindex,"dataset_name"] = "Board" 
        else:
            df_micro.at[cindex,"dataset_name"] = ""    

        df_micro.at[cindex,"llm_model"] = "& "+llms_shorname[llm]
        for k in tbl_data.keys():
            df_micro.at[cindex,k] = f"& {wins[llm][k]}"

        df_micro.at[cindex,"CatDB_test_auc_diff"] = "& "
        df_micro.at[cindex,"CatDBChain_test_auc_diff"] = "& \\\\"
    
    # for llm in llms:
    #     cindex = len(df_micro)
    #     df_micro.loc[cindex] = [None for ti in micor_tbl_cols]

    #     if llm == "gemini-1.5-pro-latest":
    #          df_micro.at[cindex,"dataset_name"] = "Leader"      
    #     else:
    #         df_micro.at[cindex,"dataset_name"] = "Board 2" 

    #     df_micro.at[cindex,"llm_model"] = "& "+llms_shorname[llm]
    #     for k in tbl_data.keys():
    #         df_micro.at[cindex,k] = f"& {wins2[llm][k]}"

    #     df_micro.at[cindex,"CatDB_test_auc_diff"] = "& "
    #     df_micro.at[cindex,"CatDBChain_test_auc_diff"] = "& \\\\"
    

    fname = f"{root_path}/tbl_micro_classification.txt"
    df_micro.to_csv(fname, index=False, header=None)
    replace_comma(fname=fname)
