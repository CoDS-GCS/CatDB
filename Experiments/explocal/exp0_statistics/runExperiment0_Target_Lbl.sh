#!/bin/bash

dataset=$1
llm_model=$2
with_dataset_description=$3
log_file_name=$4


exp_path="$(pwd)"
data_path="${exp_path}/data"

metadata_path="${data_path}/${dataset}/${dataset}.yaml"

statistic_file_name="${exp_path}/results/statistics/${dataset}.pdf"

cd "${exp_path}/setup/config"
source venv/bin/activate

SCRIPT="python DatasetLabelVisualization.py \
        --metadata-path ${metadata_path} \
        --statistic-file-name ${statistic_file_name}"

echo ${SCRIPT}
$SCRIPT