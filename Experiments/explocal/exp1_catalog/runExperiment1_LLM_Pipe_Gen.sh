#!/bin/bash

exp_path="$(pwd)"
data_path="${exp_path}/data"
dataset=$1
data_profile_path=$2
prompt_representation_type=$3
prompt_samples_type=$4
prompt_number_samples=$5
task_type=$6
llm_model=$7
with_dataset_description=$8

metadata_path="${data_path}/${dataset}/${dataset}.yaml"
number_iteration=3
number_iteration_error=7

log_file_name="${exp_path}/results/Experiment1_LLM_Pipe_Gen.dat"

output_path="${exp_path}/catdb-results/${dataset}"
mkdir -p ${output_path}

result_output_path="${exp_path}/results/Experiment1_LLM_Pipe_Gen_${prompt_representation_type}.dat"
error_output_path="${exp_path}/LLM_Pipe_Error.dat"

cd "${exp_path}/setup/Baselines/CatDB/"
source venv/bin/activate

SCRIPT="python -Wignore main.py --metadata-path ${metadata_path} \
        --data-profile-path ${data_profile_path} \
        --prompt-representation-type ${prompt_representation_type} \
        --prompt-samples-type ${prompt_samples_type} \
        --prompt-number-samples ${prompt_number_samples} \
        --prompt-number-iteration ${number_iteration} \
        --prompt-number-iteration-error ${number_iteration_error} \
        --llm-model ${llm_model} \
        --output-path ${output_path} \
        --result-output-path ${result_output_path} \
        --dataset-description ${with_dataset_description} \
        --error-output-path ${error_output_path}"

# sudo echo 3 >/proc/sys/vm/drop_caches && sudo sync
# sleep 3

echo ${SCRIPT}

start=$(date +%s%N)
$SCRIPT
end=$(date +%s%N)

echo "${dataset},${llm_model},${prompt_representation_type},${prompt_samples_type},${prompt_number_samples},${task_type},$((($end - $start) / 1000000))" >> ${log_file_name}  