#!/bin/bash

exp_path="$(pwd)"
data_path="${exp_path}/data"
dataset=$1
llm_model=$2
iteration=$3

metadata_path="${data_path}/${dataset}/${dataset}.yaml"
steps=20
result_output_path="${exp_path}/aideml-results/${dataset}"

date=$(date '+%Y-%m-%d-%H-%M-%S')
system_log="${exp_path}/system-log-${date}.dat"

mkdir -p "${exp_path}/aideml-results"
mkdir -p ${result_output_path}
result_output_path="${result_output_path}/itr-${iteration}"
rm -rf "${result_output_path}"
mkdir -p "${result_output_path}/log"
mkdir -p "${result_output_path}/workspace"


output_path="${exp_path}/results/Experiment1_LLM_AIDE.dat"

cd "${exp_path}/setup/Baselines/aideml/"
source venv/bin/activate

SCRIPT="python -Wignore run_experiment.py --metadata-path ${metadata_path} \
        --iteration ${iteration} \
        --steps ${steps} \
        --output-path ${output_path} \
        --llm-model ${llm_model} \
        --result-output-path ${result_output_path} \
        --root-data-path ${data_path} \
        --system-log ${system_log}"

echo ${SCRIPT}
$SCRIPT