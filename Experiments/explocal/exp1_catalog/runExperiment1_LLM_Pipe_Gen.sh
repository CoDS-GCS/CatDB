#!/bin/bash

exp_path="$(pwd)"
data_path="${exp_path}/data"
dataset=$1
data_profile_path=$2
prompt_representation_type=$3
prompt_example_type=$4
prompt_number_example=$5
task_type=$6
llm_model=$7

metadata_path="${data_path}/${dataset}/${dataset}.yaml"
number_iteration=10
parse_pipeline=True
run_pipeline=True

log_file_name="${exp_path}/results/Experiment1_LLM_Pipe_Gen.dat"

output_path="${exp_path}/catdb-results/${dataset}"
mkdir -p ${output_path}

result_output_path="${exp_path}/results/Experiment1_LLM_Pipe_Gen_${dataset}.dat"

cd "${exp_path}/setup/Baselines/CatDB/"
source venv/bin/activate

SCRIPT="python -Wignore main.py --metadata-path ${metadata_path} \
        --data-profile-path ${data_profile_path} \
        --prompt-representation-type ${prompt_representation_type} \
        --prompt-example-type ${prompt_example_type} \
        --prompt-number-example ${prompt_number_example} \
        --prompt-number-iteration ${number_iteration} \
        --llm-model ${llm_model} \
        --output-path ${output_path} \
        --parse-pipeline ${parse_pipeline} \
        --run-pipeline ${run_pipeline} \
        --result-output-path ${result_output_path}"

# sudo echo 3 >/proc/sys/vm/drop_caches && sudo sync
# sleep 3

echo ${SCRIPT}

start=$(date +%s%N)
$SCRIPT
end=$(date +%s%N)

echo "${data_source_name},${llm_model},${prompt_representation_type},${prompt_example_type},${prompt_number_example},${task_type},$((($end - $start) / 1000000))" >> ${log_file_name}  

echo "-----------------------------------------------------------------"