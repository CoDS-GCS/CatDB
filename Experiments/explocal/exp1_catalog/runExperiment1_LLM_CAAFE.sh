#!/bin/bash

exp_path="$(pwd)"
data_path="${exp_path}/data"
dataset=$1
llm_model=$2
with_dataset_description=$3

metadata_path="${data_path}/${dataset}/${dataset}.yaml"
number_iteration=10
result_output_path="${exp_path}/caafe-results/${dataset}"

mkdir -p "${exp_path}/caafe-results"
mkdir -p "${result_output_path}"

output_path="${exp_path}/results/Experiment1_LLM_CAAFE.dat"

cd "${exp_path}/setup/Baselines/CAAFE/"
source venv/bin/activate

SCRIPT="python -Wignore main.py --metadata-path ${metadata_path} \
        --dataset-description ${with_dataset_description} \
        --prompt-number-iteration ${number_iteration} \
        --output-path ${output_path} \
        --llm-model ${llm_model} \
        --delay 60 \
        --result-output-path ${result_output_path} \
        --data-path ${exp_path}"

echo ${SCRIPT}
$SCRIPT