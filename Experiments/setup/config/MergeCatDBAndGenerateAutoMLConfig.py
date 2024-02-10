import os
import sys
import pandas as pd
import numpy as np

def get_automl_contrain_config(max_runtime_seconds: int):
    list = [f"{max_runtime_seconds}s:",
            "  folds: 10",
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
    results_root = sys.argv[1]
    catdb_results_path = sys.argv[2]
    out_path = sys.argv[3]
    
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

    elapsed_time = ["dataset,time"]

    llms = ['gpt-3.5-turbo', 'gpt-4']

    # ds_names={"dataset_1":"Simulated Electricity",
    #           "dataset_2":"KDD98",
    #           "dataset_3":"Higgs",
    #           "dataset_4":"Airlines", 
    #           "dataset_5":"Credit g",               
    #           "dataset_6":"Microsoft",
    #           "dataset_7":"CMC", 
    #           "dataset_8":"Diabetes",
    #           "dataset_9":"Sudoku Puzzles",
    #           "dataset_10":"Pokerhand",
    #           "dataset_11":"Buzzinsocialmedia", 
    #           "dataset_12":"Zurich Transport",
    #           "dataset_13":"NYC",
    #           "dataset_14":"Black Friday",
    #           "dataset_15":"Federal Election"
    #           }

    ds_names={"horsecolic":"Horsecolic",
              "credit-g":"Credit-g",
              "albert":"Albert",
              "Sonar":"Sonar", 
              "abalone":"Abalone",               
              "poker":"Poker"
              }
    
    df_final = pd.DataFrame(columns=["dataset","llm_model","prompt_representation_type_orig","prompt_representation_type","prompt_example_type","prompt_number_example","number_iteration","task_type", "data_profile_time", "llm_pipe_gen_time", "llm_pipe_run_time", "csv_data_read_time", "total_time", "Accuracy", "F1_score", "Log_loss", "R_Squared", "RMSE"])

    extra_col_name={
        "SCHEMA":"Conf-1",
        "DISTINCT":"Conf-2",
        "MISSING_VALUE":"Conf-3",
        "NUMERIC_STATISTIC":"Conf-4",
        "CATEGORICAL_VALUE":"Conf-5",
        "DISTINCT_MISSING_VALUE":"Conf-6",
        "MISSING_VALUE_NUMERIC_STATISTIC":"Conf-7",
        "MISSING_VALUE_CATEGORICAL_VALUE":"Conf-8",
        "NUMERIC_STATISTIC_CATEGORICAL_VALUE":"Conf-9",
        "ALL":"Conf-10"
    }


    automl_max_runtime = dict()
    index = 0
    for d in os.listdir(catdb_results_path):
        print(f"------------<< {d}  >>------------------ ")
        for llm in llms:
            p = f'{catdb_results_path}/{d}/{llm}/'
            if os.path.exists(p) == False:
                continue            

            files = [f for f in os.listdir(p)]
            for f in files:
                if f in configs:
                    fnmae = f"{catdb_results_path}/{d}/{llm}/{f}"
                    
                    if os.stat(fnmae).st_size > 0: 
                        print(f)
                        results ={"Accuracy":-1, "F1_score": -1, "Log_loss":-1, "R_Squared":-1, "RMSE": -1} 
                        with open(fnmae) as fi:
                            lines = fi.readlines()                            
                            print(f"{f} >> {lines}")
                            for l in lines:

                                row = l.replace("Log loss", "Log_loss").strip().split(":")
                                if row[0] in results.keys():                                    
                                    results[row[0]] = row[1].strip()                      

                        profile_time = int(df_data_profile.at[d,"time"]) / 1000                      
                        llm_pipe_gen_time = int(df_llm_pipe_gen.at[f,"time"]) / 1000
                        try:
                            llm_pipe_run_time = int(df_llm_pipe_run.at[f, "time"]) / 1000
                        except:
                            llm_pipe_run_time = 0

                        csv_data_read_time = df_csv_data_read.at[d, "time"] / 1000
                        total_time = llm_pipe_gen_time + llm_pipe_run_time - csv_data_read_time
                        task_type = df_llm_pipe_gen.at[f, "task_type"]

                        log_names = f.replace(d+"-","").split("-")
                        prompt_representation_type = log_names[0]
                        prompt_example_type = log_names[1]
                        prompt_number_example = log_names[2]    

                        new_row = [ds_names[d], llm, prompt_representation_type, extra_col_name[prompt_representation_type],prompt_example_type, prompt_number_example,1, task_type, profile_time, llm_pipe_gen_time, llm_pipe_run_time, csv_data_read_time, total_time, results["Accuracy"], results["F1_score"], results["Log_loss"], results["R_Squared"], results["RMSE"]]

                        df_final.loc[index] = new_row
                        index +=1
                    else:
                        print(f)   
                 
                    key = f"{d}####{task_type}"
                    if key in automl_max_runtime:
                        automl_max_runtime[key] = max(total_time,  automl_max_runtime[key])
                    else:
                        automl_max_runtime[key] = total_time   

    for key in automl_max_runtime.keys():
        dataset_task = key.split("####")
        d = dataset_task[0]
        task_type = dataset_task[1]
        constrain = int(automl_max_runtime[key]+0.5)

        if constrain not in automl_contrains_set:   
            automl_contrains.append(get_automl_contrain_config(constrain))
            automl_contrains_set.add(constrain)
        
        elapsed_time.append(f"{d},{constrain}s")

    print(catdb_merge_path)
    df_final.to_csv(catdb_merge_path, index=False) 
    automl_contrains_result = "\n\n".join(automl_contrains)   
    contrains_result = "\n".join(elapsed_time)

   
    f = open(f"{out_path}/constraints_corresponding.yaml", 'w')
    f.write(automl_contrains_result)
    f.close()

    f = open(f"{out_path}/corresponding_times.csv", 'w')
    f.write(contrains_result)
    f.close()