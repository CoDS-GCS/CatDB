#!/bin/bash

dataset=$1
task_type=$2

exp_path="$(pwd)"
data_path="${exp_path}/data"
catalog_path="${exp_path}/catalog/${dataset}"


# Run Data Profiling Experiments
#./explocal/exp1_catalog/runExperiment1_Data_Profile.sh ${dataset} ${task_type}

# Rnn CSV Data Reader Experiment
# ./explocal/exp1_catalog/runExperiment1_CSVDataReader.sh $dataset

# Run Prompt and LLM Pipeline Generation Experiments
# CMD=./explocal/exp1_catalog/runExperiment1_LLM_Pipe_Gen.sh
# CMDCAAFE=./explocal/exp1_catalog/runExperiment1_LLM_CAAFE.sh
CMDAIDE=./explocal/exp1_catalog/runExperiment1_LLM_AIDE.sh
# CMDRunPipeline=./explocal/exp1_catalog/runExperiment1_Run_Local_Pipeline.sh

# $CMD ${dataset} ${catalog_path} AUTO Random 0 ${task_type} gpt-4o No
# $CMD ${dataset} ${catalog_path} AUTO Random 0 ${task_type} gemini-1.5-pro-latest No
# $CMD ${dataset} ${catalog_path} AUTO Random 0 ${task_type} llama3-70b-8192 No
# $CMD ${dataset} ${catalog_path} AUTO Random 0 ${task_type} llama-3.1-70b-versatile No


# $CMD ${dataset} ${catalog_path} CatDB Random 0 ${task_type} gpt-4o No
# $CMD ${dataset} ${catalog_path} CatDB Random 0 ${task_type} llama3-70b-8192 No
# $CMD ${dataset} ${catalog_path} CatDB Random 0 ${task_type} llama-3.1-70b-versatile No
# $CMD ${dataset} ${catalog_path} CatDB Random 0 ${task_type} mixtral-8x7b-32768 No
# $CMD ${dataset} ${catalog_path} CatDB Random 0 ${task_type} gemini-1.5-pro-latest No
# $CMD ${dataset} ${catalog_path} CatDB Random 0 ${task_type} gemini-1.5-pro-exp-0801 No
# $CMD ${dataset} ${catalog_path} CatDB Random 0 ${task_type} gemini-1.5-pro-exp-0827 No


# $CMD ${dataset} ${catalog_path} CatDBChain Random 0 ${task_type} gpt-4o No
# $CMD ${dataset} ${catalog_path} CatDBChain Random 0 ${task_type} llama3-70b-8192 No
# $CMD ${dataset} ${catalog_path} CatDBChain Random 0 ${task_type} llama-3.1-70b-versatile No
# $CMD ${dataset} ${catalog_path} CatDBChain Random 0 ${task_type} gemini-1.5-pro-exp-0801 No
#$CMD ${dataset} ${catalog_path} CatDBChain Random 0 ${task_type} gemini-1.5-pro-latest No
# $CMD ${dataset} ${catalog_path} CatDBChain Random 0 ${task_type} gemini-1.5-pro-exp-0827 No


# CAAFE
# $CMDCAAFE ${dataset} gemini-1.5-pro-latest TabPFN No
# $CMDCAAFE ${dataset} gemini-1.5-pro-latest RandomForest No

# $CMDCAAFE ${dataset} llama3-70b-8192 TabPFN No
# $CMDCAAFE ${dataset} llama3-70b-8192 RandomForest No

# $CMDCAAFE ${dataset} llama-3.1-70b-versatile TabPFN No
# $CMDCAAFE ${dataset} llama-3.1-70b-versatile RandomForest No

# $CMDCAAFE ${dataset} gpt-4o TabPFN No
# $CMDCAAFE ${dataset} gpt-4o RandomForest No


# AIDE
$CMDAIDE ${dataset} gemini-1.5-pro-latest 1
$CMDAIDE ${dataset} gemini-1.5-pro-latest 2
$CMDAIDE ${dataset} gemini-1.5-pro-latest 3
$CMDAIDE ${dataset} gemini-1.5-pro-latest 4
$CMDAIDE ${dataset} gemini-1.5-pro-latest 5
$CMDAIDE ${dataset} gemini-1.5-pro-latest 6
$CMDAIDE ${dataset} gemini-1.5-pro-latest 7
$CMDAIDE ${dataset} gemini-1.5-pro-latest 8
$CMDAIDE ${dataset} gemini-1.5-pro-latest 9
$CMDAIDE ${dataset} gemini-1.5-pro-latest 10

# Run Pipeline Localy
# $CMDRunPipeline ${dataset} CatDB Random 0 ${task_type} gemini-1.5-pro-latest No 1 "${dataset}_train" "${dataset}_test" "M"
# $CMDRunPipeline ${dataset} CatDB Random 0 ${task_type} gemini-1.5-pro-latest No 1 "${dataset}_train_clean" "${dataset}_test_clean" "G"