import pandas as pd
from MergeResults import load_merge_all_results,load_results, get_top_k_binary, get_top_k_multiclass, get_top_k_regression, mergse_dfs, get_top_k_chain , load_merde_AUTO_results


if __name__ == '__main__':
    
    root_path = "/home/saeed/Documents/Github/CatDB/Experiments/archive/SIGMOD2025-Results/"        
    df_pip_gen = load_merge_all_results(root_path=root_path)
    df_pip_gen = load_merde_AUTO_results(root_path=root_path, df = df_pip_gen)
    df_sort = pd.DataFrame(columns = df_pip_gen.columns)    
    
    datasetIDs = [("Tic-Tac-Toe","oml_dataset_6_rnc","binary",6),
                  ("Walking-Activity","oml_dataset_20_rnc","multiclass",13),
                  ("Bike-Sharing","oml_dataset_22_rnc","regression",15),
                ]
    
    datasets = []
    for ds in datasetIDs:
        ds_name, ds_rnc_name, task_type, index = ds
        #datasets.append((ds_name, ds_name, index))
        datasets.append((ds_rnc_name, f"{ds_name}",task_type , index))

    llms = ["gemini-1.5-pro-latest","llama3-70b-8192", "gpt-4o"]
    classifier = ["Auto"]

    configs = ["CatDB", "CatDBChain", "S","SDVC","SMVF","SSN","SCV","SDVCMVF","SDVCSN","SMVFSN","SMVFCV","SSNCV","ALL"]    
    ds_id = None
    for samples in {0}:
        for (ds,ds_title, task_type, index) in datasets:
            for llm in llms:
                for des in {"Yes", "No"}:                   
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
                                    df_binary = get_top_k_binary(df=df, config=config, k=1).head(1)
                                    df_binary["number_iteration"] = [ki for ki in range(1, len(df_binary)+1)]
                                    df_binary["dataset_name"] = ds_title
                                    mergse_dfs(df_sort, df_binary)

                                    # Multitask Tasks
                                    df_multi = get_top_k_multiclass(df=df, config=config, k=1).head(1)
                                    df_multi["number_iteration"] = [ki for ki in range(1, len(df_multi)+1)]
                                    df_multi["dataset_name"] = ds_title
                                    mergse_dfs(df_sort, df_multi)

                                    # Regression Tasks
                                    df_reg = get_top_k_regression(df=df, config=config, k=1).head(1)
                                    df_reg["number_iteration"] = [ki for ki in range(1, len(df_reg)+1)]
                                    df_reg["dataset_name"] = ds_title
                                    mergse_dfs(df_sort, df_reg)                                                                          
              

    df_sort.to_csv(f"{root_path}/MicroResults.csv", index=False)
          