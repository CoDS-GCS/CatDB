#!/bin/bash

dataset=$1
task_type=$2
target_attribute=$3

exp_path="$(pwd)"
data_path="${exp_path}/data"
catalog_path="${exp_path}/catalog/${dataset}"


# Run Prompt and LLM Pipeline Generation Experiments
CMD=./explocal/exp1_cleaning/runExperiment1_Clean_Categorical.sh
CMDSAGA=./explocal/exp1_cleaning/runExperiment1_SAGA.sh

#$CMD ${dataset} ${catalog_path} gemini-1.5-pro-latest
$CMDSAGA ${dataset} ${task_type}
