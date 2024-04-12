#!/bin/bash

dataset=$1
llm_model=$2
with_dataset_description=$3
log_file_name=$4

exp_path="$(pwd)"
data_path="${exp_path}/data"
data_profile_path="${data_path}/${dataset}/data_profile_full"

# Run Data Profiling
./explocal/exp0_statistics/runExperiment0_Data_Profile.sh ${dataset} ${data_profile_path}

metadata_path="${data_path}/${dataset}/${dataset}.yaml"

cd "${exp_path}/setup/Baselines/CatDB/"
source venv/bin/activate

SCRIPT="python main_statistics.py --metadata-path ${metadata_path} \
        --data-profile-path ${data_profile_path} \
        --llm-model ${llm_model} \
        --output-path ${log_file_name} \
        --dataset-description ${with_dataset_description}"

echo ${SCRIPT}
$SCRIPT