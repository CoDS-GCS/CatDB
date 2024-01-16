import os
import sys
import pandas as pd
import numpy as np
import yaml

def get_automl_contrain_config(max_runtime_seconds: int):
    list = [f"{max_runtime_seconds}:",
            "  folds: 1",
            f"  max_runtime_seconds: {max_runtime_seconds}",
            "  cores: -1",
            "  min_vol_size_mb: -1",
            "  max_mem_size_mb: -1"]
    return "\n".join(list)

def create_df_index(df):   
   
    df['log_index'] = df['dataset'] + "-" + df['prompt_representation_type']+"-"+ df['prompt_example_type']+ "-" + f'{df["prompt_number_example"]}'+"-SHOT-"+df['llm_model']+".log"

    df.set_index(df['log_index'], inplace=True)
    df.drop(columns=['log_index'], inplace=True)
    return df


if __name__ == '__main__':
    results_root = sys.argv[1]
    catdb_results_path = sys.argv[2]
    out_path = sys.argv[3]
    dataset_name = sys.argv[4]
    log_name = sys.argv[5]
    
    print(f">>>>>>>>>>>> {log_name}")
    log_names = log_name.replace(dataset_name+"-").split("-")
    prompt_representation_type = log_names[0]
    prompt_example_type = log_names[1]
    prompt_number_example = log_names[2]    
    llm = log_names[4]

    
    data_profile_result_path = f"{results_root}/Experiment1_Data_Profile.dat"
    llm_pipe_gen_result_path = f"{results_root}/Experiment1_LLM_Pipe_Gen.dat"
    llm_pipe_run_result_path = f"{results_root}/Experiment2_CatDB_LLM_Pipe_Run.dat"
    csv_data_read_result_path = f"{results_root}/Experiment1_CSVDataReader.dat"
    automl_contrains_path = f"{results_root}/constraints_corresponding.yaml"
    automl_corresponding_times = f"{results_root}/corresponding_times.csv"       
    catdb_merge_path = f"{results_root}/Experiment_CatDB_Micro_Benchmark.dat"

    df_data_profile = pd.read_csv(data_profile_result_path)
    df_data_profile.set_index(df_data_profile['dataset'], inplace=True)

    
    df_llm_pipe_gen = create_df_index(df = pd.read_csv(llm_pipe_gen_result_path))
    df_llm_pipe_run = create_df_index(df = pd.read_csv(llm_pipe_run_result_path))

    df_csv_data_read = pd.read_csv(csv_data_read_result_path)
    df_csv_data_read.set_index(df_data_profile['dataset'], inplace=True)
    

    configs = set(df_llm_pipe_gen.index)
    
    is_automl_contrains_path = os.path.isfile(automl_contrains_path)
    is_automl_corresponding_times = os.path.isfile(automl_corresponding_times)
    is_catdb_merge_path = os.path.isfile(catdb_merge_path)

    automl_contrains = ["---"]
    automl_contrains_set = set()
    
    if is_automl_contrains_path:
        with open(automl_contrains_path, "r") as f:
            try:
                contrains_data = yaml.load(f, Loader=yaml.FullLoader)

                if len(contrains_data) > 0:
                    for i in range(0, len(contrains_data)):
                        constrain = contrains_data[i]
                        automl_contrains.append(get_automl_contrain_config(constrain))
                        automl_contrains_set.add(constrain)

            except yaml.YAMLError as exc:
                raise Exception(exc)
        
        contrains_data = yaml.load(f, Loader=yaml.FullLoader)       
        

    elapsed_time = []
    if is_automl_corresponding_times:
        with open(automl_corresponding_times, "r") as f:
            row = f.readline()
            row_data = row.split(",")
            if row_data[0] != dataset_name:
                elapsed_time.append(row_data)


    if is_catdb_merge_path:
        df_final = pd.read_csv(data_profile_result_path)
    else:
        df_final = df_llm_pipe_gen.copy(deep=True)     
        df_final["data_profile_time"] = 0
        df_final["llm_pipe_gen_time"] = 0
        df_final["llm_pipe_run_time"] = 0
        df_final["csv_data_read_time"] = 0
        df_final["total_time"] = 0
        df_final["Accuracy"] = 0.0
        df_final["F1_score"] = 0.0
        df_final["Log_loss"] = 0.0
        df_final["R_Squared"] = 0.0
        df_final["RMSE"] = 0.0
        df_final = df_final.drop(log_name)
    
    automl_max_runtime = dict()

    fnmae = f"{catdb_results_path}/{dataset_name}/{llm}/{log_name}"
    if os.stat(fnmae).st_size > 0:
        results ={"Accuracy":-1, "F1_score": -1, "Log_loss":-1, "R_Squared":-1, "RMSE": -1} 
        with open(fnmae) as fi:
            
            lines = fi.readlines()
            for l in lines:
                row = l.strip().split(":")
                if row[0] in results.keys():
                    results[row[0]] = row[0].strip()

            for k in results.keys():
                df_final.at[log_name,k] = f"{results[k]}"

            profile_time = df_data_profile.at[dataset_name,"time"] / 1000                      
            llm_pipe_gen_time = df_llm_pipe_gen.at[log_name,"time"] / 1000
            llm_pipe_run_time = df_llm_pipe_run.at[log_name, "time"] / 1000
            csv_data_read_time = df_csv_data_read.at[dataset_name, "time"] / 1000
            total_time = llm_pipe_gen_time + llm_pipe_run_time - csv_data_read_time
            task_type = df_llm_pipe_gen.at[log_name, "task_type"]

            df_final.loc[log_name] = [dataset_name, llm, prompt_representation_type, prompt_example_type, prompt_number_example,1, task_type, profile_time, llm_pipe_gen_time, llm_pipe_run_time, csv_data_read_time, total_time, df_final["Accuracy", "F1_score"], df_final["Log_loss"], df_final["R_Squared"], df_final["RMSE"]]
                            
            constrain = int(max(df_final["total_time"])+0.5)
            constrain_str = f"{constrain}s"

            if constrain_str not in automl_contrains_set:   
                automl_contrains.append(get_automl_contrain_config(constrain_str))
                automl_contrains_set.add(constrain_str)

            elapsed_time.append(f"{dataset_name},{constrain}")


    df_final.to_csv(catdb_merge_path, index=False) 

    automl_contrains_result = "\n\n".join(automl_contrains)   
    contrains_result = "\n".join(elapsed_time)

   
    f = open(automl_contrains_path, 'w')
    f.write(automl_contrains_result)
    f.close()

    f = open(f"{out_path}/corresponding_times.csv", 'w')
    f.write(elapsed_time)
    f.close()