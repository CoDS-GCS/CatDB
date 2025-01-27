#!/bin/bash

dataset=$1
task_type=$2

exp_path="$(pwd)"
data_path="${exp_path}/data"
catalog_path="${exp_path}/catalog/${dataset}"


# Run Data Profiling Experiments
#./explocal/exp2_catdb/runExperiment2_Data_Profile.sh ${dataset} ${task_type}

# Rnn CSV Data Reader Experiment
# ./explocal/exp2_catdb/runExperiment2_CSVDataReader.sh $dataset

# Run Prompt and LLM Pipeline Generation Experiments
# CMD=./explocal/exp2_catdb/runExperiment2_LLM_Pipe_Gen.sh
CMDChain=./explocal/exp2_catdb/runExperiment2_LLM_Pipe_Gen_Chain.sh
# CMDRunPipeline=./explocal/exp2_catdb/runExperiment2_Run_Local_Pipeline.sh

# $CMD ${dataset} ${catalog_path} AUTO Random 0 ${task_type} gpt-4o No
# $CMD ${dataset} ${catalog_path} AUTO Random 0 ${task_type} gemini-1.5-pro-latest No
# $CMD ${dataset} ${catalog_path} AUTO Random 0 ${task_type} llama-3.1-70b-versatile No


# $CMD ${dataset} ${catalog_path} CatDB Random 0 ${task_type} gpt-4o No
# $CMD ${dataset} ${catalog_path} CatDB Random 0 ${task_type} llama-3.1-70b-versatile No
# $CMD ${dataset} ${catalog_path} CatDB Random 0 ${task_type} gemini-1.5-pro-latest No

# $CMD ${dataset} ${catalog_path} CatDBChain Random 0 ${task_type} gpt-4o No
# $CMD ${dataset} ${catalog_path} CatDBChain Random 0 ${task_type} llama-3.1-70b-versatile No
# $CMD ${dataset} ${catalog_path} CatDBChain Random 0 ${task_type} gemini-1.5-pro-latest No

$CMDChain ${dataset} ${catalog_path} CatDBChain Random 0 ${task_type} gemini-1.5-pro-latest No
#$CMDChain ${dataset} ${catalog_path} CatDBChain Random 0 ${task_type} deepseek-chat No

# Run Pipeline Localy
# $CMDRunPipeline ${dataset} CatDB Random 0 ${task_type} gemini-1.5-pro-latest No 1 "${dataset}_train" "${dataset}_test" "M"
# $CMDRunPipeline ${dataset} CatDB Random 0 ${task_type} gemini-1.5-pro-latest No 1 "${dataset}_train_clean" "${dataset}_test_clean" "G"