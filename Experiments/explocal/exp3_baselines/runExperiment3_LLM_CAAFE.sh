#!/bin/bash

exp_path="$(pwd)"
data_path="${exp_path}/data"
dataset=$1
llm_model=$2
classifier=$3
with_dataset_description=$4

metadata_path="${data_path}/${dataset}/${dataset}.yaml"
number_iteration=1
result_output_path="${exp_path}/caafe-results/${dataset}"

date=$(date '+%Y-%m-%d-%H-%M-%S')
system_log="${exp_path}/system-log-${date}.dat"

mkdir -p "${exp_path}/caafe-results"
mkdir -p "${result_output_path}"

output_path="${exp_path}/results/Experiment3_LLM_CAAFE.dat"

cd "${exp_path}/setup/Baselines/CAAFE/"
source venv/bin/activate

SCRIPT="python -Wignore main.py --metadata-path ${metadata_path} \
        --dataset-description ${with_dataset_description} \
        --prompt-number-iteration ${number_iteration} \
        --output-path ${output_path} \
        --llm-model ${llm_model} \
        --classifier ${classifier} \
        --delay 70 \
        --result-output-path ${result_output_path} \
        --data-path ${data_path} \
        --system-log ${system_log}"

echo ${SCRIPT}
$SCRIPT