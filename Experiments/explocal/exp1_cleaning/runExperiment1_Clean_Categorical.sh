#!/bin/bash

exp_path="$(pwd)"
data_path="${exp_path}/data"
dataset=$1
catalog_path=$2
llm_model=$3

metadata_path="${data_path}/${dataset}/${dataset}.yaml"
number_iteration_error=20

date=$(date '+%Y-%m-%d-%H-%M-%S')
system_log="${exp_path}/system-log-${date}.dat"

output_path="${exp_path}/catdb-results/${dataset}"
mkdir -p ${output_path}

result_output_path="${exp_path}/results/Experiment1_Clean_Categorical.dat"
error_output_path="${exp_path}/LLM_Pipe_Error.dat"

cd "${exp_path}/setup/Baselines/CatDB/"
source venv/bin/activate

SCRIPT="python -Wignore main_cleaning.py --metadata-path ${metadata_path} \
        --root-data-path ${data_path} \
        --catalog-path ${catalog_path} \
        --prompt-number-iteration-error ${number_iteration_error} \
        --llm-model ${llm_model} \
        --output-path ${output_path} \
        --result-output-path ${result_output_path} \
        --error-output-path ${error_output_path} \
        --system-log ${system_log}"


echo ${SCRIPT}

start=$(date +%s%N)
$SCRIPT
end=$(date +%s%N)