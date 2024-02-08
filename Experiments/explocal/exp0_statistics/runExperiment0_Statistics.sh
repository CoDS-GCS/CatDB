#!/bin/bash

exp_path="$(pwd)"
data_source_path="${exp_path}/data"
data_source_name=$1
prompt_representation_type=$2
suggested_model=$3
prompt_example_type=$4
prompt_number_example=$5
task_type=$6
llm_model=$7
test=$8

output_path="${exp_path}/results/statistics/${data_source_name}"

cd "${exp_path}/setup/Baselines/CatDB/"
source venv/bin/activate

SCRIPT="python main_statistics.py --data-source-path ${data_source_path} \
        --data-source-name ${data_source_name} \
        --prompt-representation-type ${prompt_representation_type} \
        --suggested-model ${suggested_model} \
        --prompt-example-type ${prompt_example_type} \
        --prompt-number-example ${prompt_number_example} \
        --llm-model ${llm_model} \
        --output-path ${output_path}"

echo ${SCRIPT}

$SCRIPT
