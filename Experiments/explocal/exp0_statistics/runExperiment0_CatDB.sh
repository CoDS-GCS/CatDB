#!/bin/bash

dataset=$1
llm_model=$2
with_dataset_description=$3
log_file_name=$4
statistic_file_name=$5
data_profile_path=$6

exp_path="$(pwd)"
data_path="${exp_path}/data"

metadata_path="${data_path}/${dataset}/${dataset}.yaml"

cd "${exp_path}/setup/Baselines/CatDB/"
source venv/bin/activate

SCRIPT="python main_statistics.py \
        --metadata-path ${metadata_path} \
        --data-profile-path ${data_profile_path} \
        --llm-model ${llm_model} \
        --dataset-description ${with_dataset_description} \
        --log-file-name ${log_file_name} \
        --statistic-file-name ${statistic_file_name} \
        --prompt-representation-type CatDB"

echo ${SCRIPT}
$SCRIPT