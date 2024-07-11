import pandas as pd
from MergeResults import  get_top_k_binary, get_top_k_multiclass, get_top_k_regression, mergse_dfs


if __name__ == '__main__':
    
    root_path = "/home/saeed/Documents/Github/CatDB/Experiments/archive/SIGMOD2025-Results/"        
    df_pip_gen = df = pd.read_csv(f"{root_path}ETE_LLM_Pipe_Gen_CatDB.dat", low_memory=False, encoding='utf-8')
    df_sort = pd.DataFrame(columns = df_pip_gen.columns)    

    df_etoe = pd.DataFrame(columns = ["dataset_name","Config","OUT","MV","NC","Result","llm_model", "has_description", "task_type","task","samples"])
    
    df_exe = pd.DataFrame(columns = ["dataset_name","dataset_name_orig", "CatDB","CatDBChain","dataset_load_time", "llm_model", "has_description", "task_type","task","samples"])    
   

    
    datasetIDs = [("Adult","gen_dataset_50","binary",1),
                  ("Bank","gen_dataset_51","binary",2),
                  ("Br2000","gen_dataset_52","binary",3)]
    
    dataset_ID = {"gen_dataset_50": "Adult", "gen_dataset_51":"Bank","gen_dataset_52":"Br2000"}
    dataset_maps = dict()

    dataset_names = df_pip_gen["dataset_name"].unique()
    all_ds = []
    for dn in dataset_names:
        dn_tmp = dn.replace("_rnc","")
        sub_names = dn_tmp.split("-")
        out = float(sub_names[2])
        np = float(sub_names[4])
        nc = int(sub_names[6])
        mv = float(sub_names[8])
        ds_dict = {"title": dataset_ID[sub_names[0]], "out": out, "np": np, "nc": nc, "mv": mv, "ismv": True}
        all_ds.append((dn, ds_dict))
       
        
    llms = ["gemini-1.5-pro-latest"] #["gemini-1.5-pro-latest","llama3-70b-8192", "gpt-4o"]
    configs = ["CatDB", "CatDBChain"] 
    classifier = ["Auto", "TabPFN", "RandomForest"]  
    for mvds, ds_dict in all_ds:
        for llm in llms:
            prompt_exe = {"CatDB": 0, "CatDBChain": 0}
            for config in configs:                    
                    tmp_df = df_pip_gen.loc[(df_pip_gen['dataset_name'] == mvds) & 
                                            (df_pip_gen['config'] == config) &
                                            (df_pip_gen['llm_model'] == llm) &
                                            (df_pip_gen['status'] == True) &
                                            (df_pip_gen['classifier'] == "Auto") &
                                            (df_pip_gen['has_description'] == "No") &
                                            (df_pip_gen['number_of_samples'] == 0)]
                                                
                    df_chain = tmp_df.loc[(tmp_df['sub_task'] == 'DataPreprocessing') | 
                                          (tmp_df['sub_task'] == 'FeatureEngineering')]
                                
                    df = tmp_df.loc[(tmp_df['sub_task'] == 'ModelSelection') | (pd.isnull(tmp_df['sub_task']))]

                    if len(df) > 0: 
                        df_sort = pd.DataFrame(columns = df_pip_gen.columns)   
                        ds_title = ds_dict["title"]                                 
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
                        df_etoe.loc[cindex] = [ds_title, config, ds_dict["out"], ds_dict["mv"], ds_dict["nc"], df_sort.iloc[0]["test_auc"], llm, "No", "binary", "classification", 0]    
                        
                        tmp_time = (df_sort['time_total'] + df_sort['time_pipeline_generate']).mean()
                        prompt_exe[config] = f"{tmp_time:.2f}"


            dataset_load_time = 1
            df_exe.loc[len(df_exe)] = [ mvds, ds_dict["title"], prompt_exe["CatDB"], prompt_exe["CatDBChain"], dataset_load_time, llm, "No", "binary", "classification", 0]

    df_etoe.to_csv(f"{root_path}/EtoEResults.csv", index=False)
    df_exe.to_csv(f"{root_path}/EtoEExeResults.csv", index=False)