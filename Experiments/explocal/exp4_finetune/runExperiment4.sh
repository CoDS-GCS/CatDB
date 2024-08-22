#!/bin/bash

dataset=$1
task_type=$2

exp_path="$(pwd)"
data_path="${exp_path}/data"
data_profile_path="${exp_path}/metadata/${dataset}/data_profile"


# Run Data Profiling Experiments
#./explocal/exp1_catalog/runExperiment1_Data_Profile.sh ${dataset} ${task_type}

# Rnn CSV Data Reader Experiment
#./explocal/exp1_catalog/runExperiment1_CSVDataReader.sh $dataset

# Run Prompt and LLM Pipeline Generation Experiments
CMD=./explocal/exp4_finetune/runExperiment4_LLM_FineTune.sh


$CMD ${dataset} ${data_profile_path} CatDB Random 0 ${task_type} gemini-1.0-pro-001 No
