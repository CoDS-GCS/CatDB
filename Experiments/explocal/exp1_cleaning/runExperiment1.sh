#!/bin/bash

dataset=$1
task_type=$2

exp_path="$(pwd)"
data_path="${exp_path}/data"
catalog_path="${exp_path}/catalog/${dataset}"


# Run Prompt and LLM Pipeline Generation Experiments
CMD=./explocal/exp1_cleaning/runExperiment1_Clean_Categorical.sh

#$CMD ${dataset} ${catalog_path} gpt-4o
# $CMD ${dataset} ${catalog_path} gpt-4-turbo
# $CMD ${dataset} ${catalog_path} llama3-70b-8192
# $CMD ${dataset} ${catalog_path} llama-3.1-70b-versatile
# $CMD ${dataset} ${catalog_path} mixtral-8x7b-32768
$CMD ${dataset} ${catalog_path} gemini-1.5-pro-latest
# $CMD ${dataset} ${catalog_path} gemini-1.5-pro-exp-0801
# $CMD ${dataset} ${catalog_path} gemini-1.5-pro-exp-0827