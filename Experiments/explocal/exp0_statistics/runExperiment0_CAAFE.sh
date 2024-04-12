#!/bin/bash

dataset=$1
llm_model=$2
with_dataset_description=$3
log_file_name=$4

exp_path="$(pwd)"
data_path="${exp_path}/data"
metadata_path="${data_path}/${dataset}/${dataset}.yaml"
number_iteration=1

cd "${exp_path}/setup/Baselines/CAAFE/"
source venv/bin/activate

SCRIPT="python CAAFEV2_Statistic.py \
        --metadata-path ${metadata_path}\
        --log-file-name ${log_file_name} \
        --number-iteration ${number_iteration} \
        --dataset-description ${with_dataset_description} \
        --llm-model ${llm_model}"

echo ${SCRIPT}
$SCRIPT
