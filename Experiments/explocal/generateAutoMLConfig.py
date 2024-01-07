import os
import pandas as pd
import numpy as np

def get_automl_contrain_config(max_runtime_seconds: int):
    list = [f"{max_runtime_seconds}s",
            "  folds: 1",
            f"  max_runtime_seconds: {max_runtime_seconds}",
            "  cores: -1",
            "  min_vol_size_mb: -1",
            "  max_mem_size_mb: -1"]
    return "\n".join(list)

def create_df_index(df):   
   
    df['log_index'] = df['dataset'] + "-" + df['prompt_representation_type']+"-"+ df['prompt_example_type']+ "-" + df['prompt_number_example']+"-SHOT-"+df['llm_model']+".log"

    df.set_index(df['log_index'], inplace=True)
    df.drop(columns=['log_index'], inplace=True)
    return df


if __name__ == '__main__':
    results_root = "../results"
    catdb_results_path = "../catdb-results"
    
    catdb_profile_path = f"{results_root}/Experiment1_Data_Profile.dat"
    catdb_llm_code_path = f"{results_root}/Experiment1_CatDB_LLM_Code.dat"
    catdb_path = f"{results_root}/Experiment1_CatDB.dat"
       
    catdb_runtime_path_final = f"{results_root}/Experiment1_CatDB_Final.dat"

    df_llm_code = create_df_index(df = pd.read_csv(catdb_llm_code_path, dtype=str))
    df_catdb = create_df_index(df = pd.read_csv(catdb_path, dtype=str))
    df_data_profile = pd.read_csv(catdb_profile_path)
    df_data_profile.set_index(df_data_profile['dataset'], inplace=True)


    configs = set(df_llm_code.index)
    
    automl_contrains = ["---"]
    automl_contrains_set = set()

    scripts = []

    llms = ['gpt-3.5-turbo', 'gpt-4']
    
    df_llm_code["data_profile_time"] = 0
    df_llm_code["pipeline_time"] = 0
    df_llm_code["catdb_llm_time"] = 0
    df_llm_code["total_time"] = 0
    df_llm_code["result"] = 0.0

    automl_max_runtime = dict()

    for d in os.listdir(catdb_results_path):
        for llm in llms:
            files = [f for f in os.listdir(f'{catdb_results_path}/{d}/{llm}/')]
            for f in files:
                if f in configs:
                    fnmae = f"{catdb_results_path}/{d}/{llm}/{f}"
                    if os.stat(fnmae).st_size > 0: 
                        with open(fnmae) as fi:
                            accuracy_result = fi.readline()   
                            raw_result = accuracy_result.split(":")
                            
                            acc = raw_result[1].strip() 
                            df_llm_code.at[f,"result"] = f"{acc}"

                            profile_time = int(df_data_profile.at[d,"time"]) / 1000
                            df_llm_code.at[f, "data_profile_time"] = f"{profile_time}"
                            
                            pipline_time = int(df_llm_code.at[f,"time"]) / 1000
                            df_llm_code.at[f,"pipeline_time"] = pipline_time

                            llm_time = int(df_catdb.at[f, "time"]) / 1000
                            df_llm_code.at[f, "catdb_llm_time"] = f"{llm_time}"

                            total_time = profile_time + pipline_time + llm_time
                            df_llm_code.at[f, "total_time"] = total_time                            

                            task_type = df_llm_code.at[f, "task_type"]
                            key = f"{d}####{task_type}"
                            if key in automl_max_runtime:
                                automl_max_runtime[key] = max(total_time,  automl_max_runtime[key])
                            else:
                                 automl_max_runtime[key] = total_time   

    for key in automl_max_runtime.keys():
        dataset_task = key.split("####")
        d = dataset_task[0]
        task_type = dataset_task[1]
        constrain = int(automl_max_runtime[key] + 5)

        if constrain not in automl_contrains_set:   
            automl_contrains.append(get_automl_contrain_config(constrain))
            automl_contrains_set.add(constrain)
        
        scripts.append(f"./explocal/exp1_systematic/runExperiment1.sh {d} {task_type} {constrain}s")

    df_llm_code.to_csv(catdb_runtime_path_final, index=False) 
    automl_contrains_result = "\n\n".join(automl_contrains)   
    scripts_result = "\n".join(scripts)

   
    f = open("exp1_systematic/automl_config/constraints_catdb.yaml", 'w')
    f.write(automl_contrains_result)
    f.close()

    f = open("exp1_systematic/automl_config/scripts.sh", 'w')
    f.write(scripts_result)
    f.close()