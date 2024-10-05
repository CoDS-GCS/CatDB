import pandas as pd
from MergeResults import load_merge_all_results, get_top_k_regression, replace_comma, dataset_corr


if __name__ == '__main__':
    
    root_path = "/home/saeed/Documents/Github/CatDB/Experiments/archive/SIGMOD2025-Results/"        
    df_pip_gen = load_merge_all_results(root_path=root_path)
    
    micor_tbl_cols = ["dataset_name","llm_model",
                      "CatDB_train_r_squared","CatDB_test_r_squared",
                      "CatDBChain_train_r_squared","CatDBChain_test_r_squared",                                       
                      "AutoSklearn_train_r_squared","AutoSklearn_test_r_squared",
                      "H2O_train_r_squared","H2O_test_r_squared",
                      "Flaml_train_r_squared","Flaml_test_r_squared",
                      "Autogluon_train_r_squared","Autogluon_test_r_squared",
                      "CatDB_test_r_squared_diff","CatDBChain_test_r_squared_diff"]
    df_micro = pd.DataFrame(columns = micor_tbl_cols)   
    df_overall = pd.DataFrame(columns = ["dataset_name","llm_model","CatDB","CatDBChain","max_other"])
    
    datasetIDs = [("Black-Friday","oml_dataset_21_rnc",15),
                  ("Bike-Sharing","oml_dataset_22_rnc",16),
                  ("NYC","oml_dataset_24_rnc",17),
                  ("House-Sales","oml_dataset_23_rnc",18),                  
                  #("Airlines-DepDelay","oml_dataset_25_rnc",18)
                  ]
    datasets = []
    for ds in datasetIDs:
        ds_name, ds_rnc_name, index = ds
        datasets.append((ds_rnc_name, f"{ds_name}", index))

    
    wins = { "gemini-1.5-pro-latest":{ "CatDB_train_r_squared":0,"CatDB_test_r_squared":0,
                        "CatDBChain_train_r_squared":0,"CatDBChain_test_r_squared":0,                        
                        "AutoSklearn_train_r_squared":0,"AutoSklearn_test_r_squared":0,
                        "H2O_train_r_squared":0,"H2O_test_r_squared":0,
                        "Flaml_train_r_squared":0,"Flaml_test_r_squared":0,
                        "Autogluon_train_r_squared":0,"Autogluon_test_r_squared":0},
            "llama3-70b-8192":{"CatDB_train_r_squared":0,"CatDB_test_r_squared":0,
                        "CatDBChain_train_r_squared":0,"CatDBChain_test_r_squared":0,                        
                        "AutoSklearn_train_r_squared":0,"AutoSklearn_test_r_squared":0,
                        "H2O_train_r_squared":0,"H2O_test_r_squared":0,
                        "Flaml_train_r_squared":0,"Flaml_test_r_squared":0,
                        "Autogluon_train_r_squared":0,"Autogluon_test_r_squared":0},
            "gpt-4o":{"CatDB_train_r_squared":0,"CatDB_test_r_squared":0,
                        "CatDBChain_train_r_squared":0,"CatDBChain_test_r_squared":0,                        
                        "AutoSklearn_train_r_squared":0,"AutoSklearn_test_r_squared":0,
                        "H2O_train_r_squared":0,"H2O_test_r_squared":0,
                        "Flaml_train_r_squared":0,"Flaml_test_r_squared":0,
                        "Autogluon_train_r_squared":0,"Autogluon_test_r_squared":0}            }
    

    #llms = ["gpt-4", "gpt-4o", "llama3-70b-8192", "gemini-1.5-pro-latest"]
    llms = ["gpt-4o", "gemini-1.5-pro-latest","llama3-70b-8192",]
    llms_shorname = {"gpt-4o":"GPT-4o","gemini-1.5-pro-latest":"Gemini-1.5","llama3-70b-8192":"Llama3-70b"}

    configs = ["CatDB", "CatDBChain", "CAAFE","AutoSklearn","H2O","Flaml","Autogluon"]    
    ds_id = None
    for (ds,ds_title, index) in datasets:
        for llm in llms:
            tbl_data = {"CatDB_train_r_squared":0,"CatDB_test_r_squared":0,
                        "CatDBChain_train_r_squared":0,"CatDBChain_test_r_squared":0,                        
                        "AutoSklearn_train_r_squared":0,"AutoSklearn_test_r_squared":0,
                        "H2O_train_r_squared":0,"H2O_test_r_squared":0,
                        "Flaml_train_r_squared":0,"Flaml_test_r_squared":0,
                        "Autogluon_train_r_squared":0,"Autogluon_test_r_squared":0}
            for des in {"Yes", "No"}:                
                for config in configs:               
                            df = df_pip_gen.loc[(df_pip_gen['dataset_name'] == ds) & 
                                                (df_pip_gen['config'] == config) &
                                                (df_pip_gen['llm_model'] == llm) &
                                                (df_pip_gen['status'] == True) &
                                                (df_pip_gen['has_description'] == des) &
                                                ((df_pip_gen['sub_task'] == 'ModelSelection') | (pd.isnull(df_pip_gen['sub_task'])))]
                            
                            if config in {"AutoSklearn","H2O","Flaml","Autogluon"}:
                               df = df.loc[df['number_iteration'] == 1].head(1)                               

                            if len(df) > 0:                             
                                
                                # Regression Tasks
                                df_ds = get_top_k_regression(df=df, config=config, k=1)
                                df_ds["number_iteration"] = [ki for ki in range(1, len(df_ds)+1)]     
                                

                                col_train = "train_r_squared"
                                col_test = "test_r_squared"
                                
                                key_train = f"{config}_train_r_squared"
                                key_test =  f"{config}_test_r_squared"

                                if len(df_ds) == 0:
                                    tbl_data[key_train] = None
                                    tbl_data[key_test] = None
                                    continue

                                tbl_data[key_train] = int(df_ds.iloc[0][col_train] * 10000)
                                tbl_data[key_test] = int(df_ds.iloc[0][col_test] * 10000)
                                

            max_train = 0
            max_test = 0
            max_other = 0
            for k in tbl_data.keys():
                r2_value =  tbl_data[k]
                if r2_value == None:
                    r2_value = 0
                if "test" in k:
                    max_test = max(max_test, r2_value)
                    
                    if "CatDB" not in k:
                        max_other = max(max_other, r2_value)

                elif "train" in k:    
                    max_train = max(max_train, r2_value) 

            
            cindex = len(df_micro)
            df_micro.loc[cindex] = [None for ti in micor_tbl_cols]
            
            tbl_line = "" #"\\cmidrule{2-16}"
            if llm == "gpt-4o":
             df_micro.at[cindex,"dataset_name"] = "\multirow{3}{*}{"+ds_title+"}"         
            else:
                df_micro.at[cindex,"dataset_name"] = "" 
            
            if llm == "llama3-70b-8192":    
                tbl_line = "\\chline"
                
            df_micro.at[cindex,"llm_model"] = "& "+llms_shorname[llm]
            for k in tbl_data.keys():
                htext = ""
                corr_config = dataset_corr[ds]               
                if (corr_config in {"CatDB"} and k in {"CatDB_train_r_squared", "CatDB_test_r_squared"}) or (corr_config in {"CatDBChain"} and k in {"CatDBChain_train_r_squared", "CatDBChain_test_r_squared"}):
                        htext="\\cellcolor{lightgray!50}" 

                if tbl_data[k] is None:
                    df_micro.at[cindex,k] = "& N/A"   
                elif "test" in k and tbl_data[k] >= max_test:
                    df_micro.at[cindex,k] = "&"+htext+" \\textbf{"+f"{tbl_data[k]/100}"+"}"
                    wins[llm][k] +=1
                elif "train" in k and tbl_data[k] >= max_train:
                    df_micro.at[cindex,k] = "&"+htext+" \\textbf{"+f"{tbl_data[k]/100}"+"}"
                    wins[llm][k] +=1  
                elif tbl_data[k] == 0:
                    if ds_title in {"House-Sales"} and k in {"AutoSklearn_test_r_squared","AutoSklearn_train_r_squared"}:
                            if llm=="gemini-1.5-pro-latest":    
                                if k == "AutoSklearn_train_r_squared":                                
                                    df_micro.at[cindex,k] = "& \multicolumn{2}{c|}{Out of Time}"
                                else:
                                  df_micro.at[cindex,k] = ""     
                            else:
                             df_micro.at[cindex,k] = "&"  
                    elif "train_r_squared" in k:       
                      df_micro.at[cindex,k] = "& \multicolumn{2}{c|}{Out of Time}"
                    else:
                       df_micro.at[cindex,k] = "" 
                else:
                    df_micro.at[cindex,k] = f"& {htext} {tbl_data[k]/100}"
            
            catdb_value =(tbl_data["CatDB_test_r_squared"]-max_other)/100
            catdb_chain_value = (tbl_data["CatDBChain_test_r_squared"]-max_other) / 100
            
            if -1< catdb_value <1:
                    catdb_value_str = f"\char`\~\ 0"

            elif catdb_value <0:
                catdb_value_str = f"${catdb_value}$"
            else:
               catdb_value_str = f"$+{catdb_value}$"

            if -1< catdb_chain_value <1:
                    catdb_chain_value_str = f"\char`\~\ 0"  

            elif catdb_chain_value <0:
                catdb_chain_value_str = f"${catdb_chain_value}$"
            else:
                catdb_chain_value_str = f"$+{catdb_chain_value}$"

            df_micro.at[cindex,"CatDB_test_r_squared_diff"] = f'& {catdb_value_str}'
            df_micro.at[cindex,"CatDBChain_test_r_squared_diff"] = f'& {catdb_chain_value_str} \\\\ {tbl_line}'

            df_overall.loc[len(df_overall)] = [ds_title, llm, tbl_data["CatDB_test_r_squared"] /100, tbl_data["CatDBChain_test_r_squared"] /100, max_other/100]

            
    # add leader board:
    for llm in llms:
        cindex = len(df_micro)
        df_micro.loc[cindex] = [None for ti in micor_tbl_cols]

        if llm == "gpt-4o":
             df_micro.at[cindex,"dataset_name"] = "\multirow{3}{*}{Leader Board}"      
        else:
            df_micro.at[cindex,"dataset_name"] = "" 

        df_micro.at[cindex,"llm_model"] = "& "+llms_shorname[llm]
        for k in tbl_data.keys():
            df_micro.at[cindex,k] = f"& {wins[llm][k]}"

        df_micro.at[cindex,"CatDB_test_r_squared_diff"] = "& "
        df_micro.at[cindex,"CatDBChain_test_r_squared_diff"] = "& \\\\"
    

    fname = f"{root_path}/tbl_micro_regression.txt"
    df_micro.to_csv(fname, index=False, header=None)
    replace_comma(fname=fname)

    df_overall.to_csv(f"{root_path}/OverallResults_Regression.csv", index=False)
    