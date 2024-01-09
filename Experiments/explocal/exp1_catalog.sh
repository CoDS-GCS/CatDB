#!/bin/bash


exp_path="$(pwd)"

echo "dataset,platform,time" >> "${exp_path}/results/Experiment1_Data_Profile.dat"
echo "dataset,time" >> "${exp_path}/results/Experiment1_CSVDataReader.dat"
echo "dataset,llm_model,prompt_representation_type,prompt_example_type,prompt_number_example,prompt_number_iteration,task_type,time,result" >> "${exp_path}/results/Experiment1_LLM_Pipe_Gen.dat"

CMD=./explocal/exp1_systematic/runExperiment1.sh

$CMD airlines binary 