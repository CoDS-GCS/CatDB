#!/bin/bash

dataset=$1
task_type=$2

exp_path="$(pwd)"
data_path="${exp_path}/data"
data_profile_path="${exp_path}/metadata/${dataset}/data_profile"


# Run Prompt and LLM Pipeline Generation Experiments
CMD=./explocal/exp5_dataprepare/runExperiment5_LLM_MissingValue.sh


$CMD ${dataset} ${data_profile_path} CatDB Random 50 100 ${task_type} gemini-1.5-pro-exp-0827 No
