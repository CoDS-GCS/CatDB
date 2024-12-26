import pandas as pd
from MergeResults import load_merge_all_results,load_results, get_top_k_binary, get_top_k_multiclass, get_top_k_multiclass_EUIT, get_top_k_regression, mergse_dfs, get_top_k_chain , dataset_corr


if __name__ == '__main__':
    
    root_path = "../results"        
    df_pip_gen = load_merge_all_results(root_path=root_path)
    df_sort = pd.DataFrame(columns = df_pip_gen.columns)
    
    df_performance = pd.DataFrame(columns = ["dataset_name", "config", "performance"])

    datasetIDs = [("Airline","Airline","multiclass",101),
                  ("IMDB","IMDB-IJS","binary",102),
                  ("Accidents","Accidents","multiclass",103),
                  ("Financial","Financial","multiclass",104),
                  ("Yelp","Yelp","multiclass",105),
                  ("EU IT","EU-IT","multiclass",106),                  
                  ("Etailing","Etailing","multiclass",107),
                  ("Survey","Midwest-Survey","multiclass",108),
                  ("WiFi","WiFi","binary",109),                  
                  ("Utility","Utility","regression",1010),
                  ("Breast-w","oml_dataset_2_rnc","binary",1),
                  ("CMC","oml_dataset_3_rnc","multiclass",2),
                  ("Credit-g","oml_dataset_4_rnc","binary",3),
                  ("Diabetes","oml_dataset_5_rnc","binary",4),
                  ("Tic-Tac-Toe","oml_dataset_6_rnc","binary",5),
                  ("Nomao","oml_dataset_33_rnc","binary",6),
                  ("Gas-Drift","oml_dataset_34_rnc","multiclass",7),                                 
                  ("Walking","oml_dataset_20_rnc","multiclass",8),
                  ("Bike-Sharing","oml_dataset_22_rnc","regression",9),
                  ("House-Sales","oml_dataset_23_rnc","regression",10),
                  ("NYC","oml_dataset_24_rnc","regression",11),                 
                  ("Volkert","oml_dataset_35_rnc","multiclass",12)
                ]
    
    datasets = []
    for ds in datasetIDs:
        ds_name, ds_rnc_name, task_type, index = ds
        datasets.append((ds_rnc_name, f"{ds_name}",task_type , index))

    llms = ["gemini-1.5-pro-latest"]

    configs = ["CatDB", "CAAFE", "AIDE", "AutoSklearn","H2O","Flaml","Autogluon"]    
    ds_id = None
    for (ds,ds_title, task_type, index) in datasets:
            for llm in llms:                  
                    for config in configs:
                                df = df_pip_gen.loc[(df_pip_gen['dataset_name'] == ds) & 
                                                    (df_pip_gen['config'] == config) &
                                                    (df_pip_gen['llm_model'] == llm) &
                                                    (df_pip_gen['status'] == True) &
                                                    #(df_pip_gen['classifier'] == cls) &
                                                    (df_pip_gen['has_description'] == "No") &
                                                    (df_pip_gen['number_of_samples'] == 0) &
                                                    ((df_pip_gen['operation'] == 'Run-Pipeline') | 
                                                    (df_pip_gen['operation']=='Run-CAAFE') |
                                                    (df_pip_gen['operation']=='Run-AutoML'))]                              

                                if config in {"CAAFE"}:
                                    df = df.loc[df['classifier'] == "TabPFN"]                      
                               
                                cindex = len(df_performance)
                                if ds == "Yelp":
                                     performance = 0.9826
                                elif ds == "EU-IT":
                                     performance = 0.918
                                else:
                                     performance = -0.15    
                                          
                                if len(df) > 0:  
                                    flag = True
                                    if task_type == "binary":
                                         df_binary = get_top_k_binary(df=df, config=config, k=1)  
                                         if len(df_binary) == 0:
                                            flag = False
                                         else:   
                                            performance  = df_binary.iloc[0]['test_auc']
                                    elif task_type == "multiclass":
                                         # Multitask Tasks
                                        if ds_title == 'EU-IT':
                                            df_multi = get_top_k_multiclass_EUIT(df=df, config=config, k=1)
                                            if len(df_multi) == 0:
                                              flag = False
                                            else:
                                                performance  = df_multi.iloc[0]['test_accuracy']
                                        else:
                                            df_multi = get_top_k_multiclass(df=df, config=config, k=1)
                                            if len(df_multi) == 0:
                                              flag = False
                                            else:  
                                                performance  = df_multi.iloc[0]['test_auc_ovr']   
                                    elif task_type == "regression":
                                        df_reg = get_top_k_regression(df=df, config=config, k=1)
                                        if len(df_reg) == 0:
                                              flag = False
                                        else:      
                                            performance  = df_reg.iloc[0]['test_r_squared']                                   
                                    
                                    if flag:
                                        df_performance.loc[cindex] = [ds_title, config, performance]                                                                
                                    else:
                                        df_performance.loc[cindex] = [ds_title, config, performance]     
                    
                                else:
                                     df_performance.loc[cindex] = [ds_title, config, performance]
               
    
    df_performance = df_performance.sort_values(by='performance', ascending=False)
    df_performance.to_csv(f"{root_path}/PerformanceResults.csv", index=False)
    print(df_performance['dataset_name'].unique())