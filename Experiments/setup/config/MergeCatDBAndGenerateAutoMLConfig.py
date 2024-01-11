import os
import sys
import pandas as pd
import numpy as np

def get_automl_contrain_config(max_runtime_seconds: int):
    list = [f"{max_runtime_seconds}s:",
            "  folds: 1",
            f"  max_runtime_seconds: {max_runtime_seconds}",
            "  cores: -1",
            "  min_vol_size_mb: -1",
            "  max_mem_size_mb: -1"]
    return "\n".join(list)

def get_automl_corresponding_script(max_runtime_seconds: int):
    list = ["#!/bin/bash",
            "task_type=$1",
            "\n",
            ""]
    return "\n".join(list)

def create_df_index(df):   
   
    df['log_index'] = df['dataset'] + "-" + df['prompt_representation_type']+"-"+ df['prompt_example_type']+ "-" + df['prompt_number_example']+"-SHOT-"+df['llm_model']+".log"

    df.set_index(df['log_index'], inplace=True)
    df.drop(columns=['log_index'], inplace=True)
    return df


if __name__ == '__main__':
    results_root = sys.argv[1]
    catdb_results_path = sys.argv[2]
    
    data_profile_result_path = f"{results_root}/Experiment1_Data_Profile.dat"
    llm_pipe_gen_result_path = f"{results_root}/Experiment1_LLM_Pipe_Gen.dat"
    llm_pipe_run_result_path = f"{results_root}/Experiment2_CatDB_LLM_Pipe_Run.dat"
    csv_data_read_result_path = f"{results_root}/Experiment1_CSVDataReader.dat"
       
    catdb_merge_path = f"{results_root}/Experiment_CatDB_Micro_Benchmark.dat"

    df_data_profile = pd.read_csv(data_profile_result_path)
    df_data_profile.set_index(df_data_profile['dataset'], inplace=True)

    
    df_llm_pipe_gen = create_df_index(df = pd.read_csv(llm_pipe_gen_result_path, dtype=str))
    df_llm_pipe_run = create_df_index(df = pd.read_csv(llm_pipe_run_result_path, dtype=str))

    df_csv_data_read = pd.read_csv(csv_data_read_result_path)
    df_csv_data_read.set_index(df_data_profile['dataset'], inplace=True)
    

    configs = set(df_llm_pipe_gen.index)
    
    automl_contrains = ["---"]
    automl_contrains_set = set()

    scripts = []

    llms = ['gpt-3.5-turbo', 'gpt-4']
    
    df_llm_pipe_gen["data_profile_time"] = 0
    df_llm_pipe_gen["llm_pipe_gen_time"] = 0
    df_llm_pipe_gen["llm_pipe_run_time"] = 0
    df_llm_pipe_gen["csv_data_read_time"] = 0
    df_llm_pipe_gen["total_time"] = 0
    df_llm_pipe_gen["result"] = 0.0

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
                            
                            res = raw_result[1].strip() 
                            df_llm_pipe_gen.at[f,"result"] = f"{res}"

                            profile_time = int(df_data_profile.at[d,"time"]) / 1000
                            df_llm_pipe_gen.at[f, "data_profile_time"] = f"{profile_time}"
                            
                            llm_pipe_gen_time = int(df_llm_pipe_gen.at[f,"time"]) / 1000
                            df_llm_pipe_gen.at[f,"llm_pipe_gen_time"] = llm_pipe_gen_time

                            llm_pipe_run_time = int(df_llm_pipe_run.at[f, "time"]) / 1000
                            df_llm_pipe_gen.at[f, "llm_pipe_run_time"] = f"{llm_pipe_run_time}"

                            csv_data_read_time = int(df_csv_data_read.at[d, "time"]) / 1000
                            df_llm_pipe_gen.at[f, "csv_data_read_time"] = f"{csv_data_read_time}"

                            total_time = llm_pipe_gen_time + llm_pipe_run_time - csv_data_read_time
                            df_llm_pipe_gen.at[f, "total_time"] = total_time                            

                            task_type = df_llm_pipe_gen.at[f, "task_type"]
                            
                            key = f"{d}####{task_type}"
                            if key in automl_max_runtime:
                                automl_max_runtime[key] = max(total_time,  automl_max_runtime[key])
                            else:
                                 automl_max_runtime[key] = total_time   

    for key in automl_max_runtime.keys():
        dataset_task = key.split("####")
        d = dataset_task[0]
        task_type = dataset_task[1]
        constrain = int(automl_max_runtime[key])

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