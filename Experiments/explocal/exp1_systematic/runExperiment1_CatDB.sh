#!/bin/bash

exp_path="$(pwd)"
data_source_path="${exp_path}/data"
data_source_name=$1
prompt_representation_type=$2
prompt_example_type=$3
prompt_number_example=$4
prompt_number_iteration=$5
task_type=$6
log_file_name=$7
llm_model=$8

output_path="${exp_path}/llm-results/${llm_model}"
mkdir -p ${output_path}

cd "${exp_path}/setup/Baselines/CatDB/"
source venv/bin/activate

SCRIPT="python main.py --data-source-path ${data_source_path} \
        --data-source-name ${data_source_name} \
        --prompt-representation-type ${prompt_representation_type} \
        --prompt-example-type ${prompt_example_type} \
        --prompt-number-example ${prompt_number_example} \
        --prompt-number-iteration ${prompt_number_iteration} \
        --llm-model ${llm_model} \
        --output-path ${output_path}"

# sudo echo 3 >/proc/sys/vm/drop_caches && sudo sync
# sleep 3

echo ${SCRIPT}

start=$(date +%s%N)
$SCRIPT
end=$(date +%s%N)

echo "${data_source_name},${llm_model},${prompt_representation_type},${prompt_example_type},${prompt_number_example},${prompt_number_iteration},${task_type},$((($end - $start) / 1000000))" >> $log_file_name