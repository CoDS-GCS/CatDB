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
CMD=./explocal/exp1_catalog/runExperiment1_LLM_Pipe_Gen.sh
#CMDRun=./explocal/exp1_catalog/runExperiment1_Run_Pipe.sh
#CMDCAAFE=./explocal/exp1_catalog/runExperiment1_LLM_CAAFE.sh

# $CMD ${dataset} ${data_profile_path} AUTO Random 0 ${task_type} gpt-4o No
# $CMD ${dataset} ${data_profile_path} AUTO Random 0 ${task_type} gemini-1.5-pro-latest No
# $CMD ${dataset} ${data_profile_path} AUTO Random 0 ${task_type} llama3-70b-8192 No
# $CMD ${dataset} ${data_profile_path} AUTO Random 0 ${task_type} llama-3.1-70b-versatile No


# $CMD ${dataset} ${data_profile_path} CatDB Random 0 ${task_type} gpt-4o No
# $CMD ${dataset} ${data_profile_path} CatDB Random 0 ${task_type} llama3-70b-8192 No
# $CMD ${dataset} ${data_profile_path} CatDB Random 0 ${task_type} llama-3.1-70b-versatile No
# $CMD ${dataset} ${data_profile_path} CatDB Random 0 ${task_type} gemini-1.5-pro-latest No
# $CMD ${dataset} ${data_profile_path} CatDB Random 0 ${task_type} gemini-1.5-pro-exp-0801 No


# $CMD ${dataset} ${data_profile_path} CatDBChain Random 0 ${task_type} gpt-4o No
# $CMD ${dataset} ${data_profile_path} CatDBChain Random 0 ${task_type} llama3-70b-8192 No
# $CMD ${dataset} ${data_profile_path} CatDBChain Random 0 ${task_type} llama-3.1-70b-versatile No
$CMD ${dataset} ${data_profile_path} CatDBChain Random 0 ${task_type} gemini-1.5-pro-latest No


# CAAFE
# $CMDCAAFE ${dataset} gemini-1.5-pro-latest TabPFN No
# $CMDCAAFE ${dataset} gemini-1.5-pro-latest RandomForest No

# $CMDCAAFE ${dataset} llama3-70b-8192 TabPFN No
# $CMDCAAFE ${dataset} llama3-70b-8192 RandomForest No

# $CMDCAAFE ${dataset} llama-3.1-70b-versatile TabPFN No
# $CMDCAAFE ${dataset} llama-3.1-70b-versatile RandomForest No

# $CMDCAAFE ${dataset} gpt-4o TabPFN No
# $CMDCAAFE ${dataset} gpt-4o RandomForest No