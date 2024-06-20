#!/bin/bash

dataset=$1
task_type=$2

exp_path="$(pwd)"
data_path="${exp_path}/data"
data_profile_path="${data_path}/${dataset}/data_profile"


# Run Data Profiling Experiments
#./explocal/exp1_catalog/runExperiment1_Data_Profile.sh ${dataset} ${task_type}

# Rnn CSV Data Reader Experiment
#./explocal/exp1_catalog/runExperiment1_CSVDataReader.sh $dataset

# Run Prompt and LLM Pipeline Generation Experiments
CMD=./explocal/exp1_catalog/runExperiment1_LLM_Pipe_Gen.sh
#CMDCAAFE=./explocal/exp1_catalog/runExperiment1_LLM_CAAFE.sh

# $CMD ${dataset} ${data_profile_path} AUTO Random 0 ${task_type} gpt-4 Yes
# $CMD ${dataset} ${data_profile_path} AUTO Random 0 ${task_type} gpt-4o Yes
# $CMD ${dataset} ${data_profile_path} AUTO Random 0 ${task_type} gpt-3.5-turbo Yes
# $CMD ${dataset} ${data_profile_path} AUTO Random 0 ${task_type} llama3-70b-8192 Yes

# $CMD ${dataset} ${data_profile_path} AUTO Random 0 ${task_type} gpt-4 No
# $CMD ${dataset} ${data_profile_path} AUTO Random 0 ${task_type} gpt-4o No
# $CMD ${dataset} ${data_profile_path} AUTO Random 0 ${task_type} gpt-3.5-turbo No
# $CMD ${dataset} ${data_profile_path} AUTO Random 0 ${task_type} llama3-70b-8192 No

# $CMD ${dataset} ${data_profile_path} CatDB Random 0 ${task_type} gpt-4 Yes
# $CMD ${dataset} ${data_profile_path} CatDB Random 0 ${task_type} gpt-4o Yes
# $CMD ${dataset} ${data_profile_path} CatDB Random 0 ${task_type} gpt-3.5-turbo Yes

# $CMD ${dataset} ${data_profile_path} CatDB Random 0 ${task_type} llama3-8b-8192 Yes
# $CMD ${dataset} ${data_profile_path} CatDB Random 0 ${task_type} gemma-7b-it Yes
# $CMD ${dataset} ${data_profile_path} CatDB Random 0 ${task_type} mixtral-8x7b-32768 Yes
# $CMD ${dataset} ${data_profile_path} CatDB Random 0 ${task_type} llama3-70b-8192 Yes
$CMD ${dataset} ${data_profile_path} CatDB Random 0 ${task_type} gemini-1.5-pro-latest Yes

# $CMD ${dataset} ${data_profile_path} CatDB Random 0 ${task_type} gpt-4 No
# $CMD ${dataset} ${data_profile_path} CatDB Random 0 ${task_type} gpt-4o No
# $CMD ${dataset} ${data_profile_path} CatDB Random 0 ${task_type} gpt-3.5-turbo No
# $CMD ${dataset} ${data_profile_path} CatDB Random 0 ${task_type} llama3-70b-8192 No

#$CMD ${dataset} ${data_profile_path} CatDBChain Random 0 ${task_type} gpt-4 Yes
# $CMD ${dataset} ${data_profile_path} CatDBChain Random 0 ${task_type} gpt-4o Yes
# $CMD ${dataset} ${data_profile_path} CatDBChain Random 0 ${task_type} gpt-3.5-turbo Yes

# $CMD ${dataset} ${data_profile_path} CatDBChain Random 0 ${task_type} llama3-8b-8192 Yes
# $CMD ${dataset} ${data_profile_path} CatDBChain Random 0 ${task_type} gemma-7b-it Yes
# $CMD ${dataset} ${data_profile_path} CatDBChain Random 0 ${task_type} mixtral-8x7b-32768 Yes
# $CMD ${dataset} ${data_profile_path} CatDBChain Random 0 ${task_type} llama3-70b-8192 Yes
# $CMD ${dataset} ${data_profile_path} CatDBChain Random 0 ${task_type} gemini-1.5-pro-latest Yes

# $CMD ${dataset} ${data_profile_path} CatDBChain Random 0 ${task_type} gpt-4 No
# $CMD ${dataset} ${data_profile_path} CatDBChain Random 0 ${task_type} gpt-4o No
# $CMD ${dataset} ${data_profile_path} CatDBChain Random 0 ${task_type} gpt-3.5-turbo No
# $CMD ${dataset} ${data_profile_path} CatDBChain Random 0 ${task_type} llama3-70b-8192 No
# $CMD ${dataset} ${data_profile_path} CatDBChain Random 0 ${task_type} gemini-1.5-pro-latest No

# Run Generated Pipeline
# CMD=./explocal/exp1_catalog/runExperiment1_Run_Pipe.sh

# $CMD ${dataset} ${data_profile_path} CatDB Random 0 ${task_type} gemini-1.5-pro-latest Yes 3
# $CMD ${dataset} ${data_profile_path} CatDB Random 0 ${task_type} gemini-1.5-pro-latest Yes 4

# CAAFE
#$CMDCAAFE ${dataset} gemini-1.5-pro-latest TabPFN Yes
#$CMDCAAFE ${dataset} gemini-1.5-pro-latest RandomForest Yes