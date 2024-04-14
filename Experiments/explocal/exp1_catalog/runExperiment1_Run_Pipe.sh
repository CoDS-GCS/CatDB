#!/bin/bash

exp_path="$(pwd)"
data_path="${exp_path}/data"
dataset=$1
prompt_representation_type=$2
prompt_example_type=$3
prompt_number_example=$4
task_type=$5
llm_model=$6
with_dataset_description=$7

metadata_path="${data_path}/${dataset}/${dataset}.yaml"
number_iteration=20
parse_pipeline=True
run_pipeline=True

log_file_name="${exp_path}/results/Experiment1_Run_Pipe.dat"

output_path="${exp_path}/catdb-results/${dataset}"
mkdir -p ${output_path}

result_output_path="${exp_path}/results/Experiment1_Run_Pipe_${prompt_representation_type}.dat"

cd "${exp_path}/setup/Baselines/CatDB/"
source venv/bin/activate

src_path="${output_path}/${llm_model}-${prompt_representation_type}-${prompt_example_type}-${prompt_number_example}-SHOT-${with_dataset_description}.py"

SCRIPT="python -Wignore main_run_code.py --metadata-path ${metadata_path} \
        --prompt-representation-type ${prompt_representation_type} \
        --prompt-example-type ${prompt_example_type} \
        --prompt-number-example ${prompt_number_example} \
        --prompt-number-iteration ${number_iteration} \
        --llm-model ${llm_model} \
        --output-path ${output_path} \
        --parse-pipeline ${parse_pipeline} \
        --run-pipeline ${run_pipeline} \
        --result-output-path ${result_output_path} \
        --dataset-description ${with_dataset_description} \
        --src-path ${src_path}"

echo ${SCRIPT}

start=$(date +%s%N)
$SCRIPT
end=$(date +%s%N)

echo "${dataset},${llm_model},${prompt_representation_type},${prompt_example_type},${prompt_number_example},${task_type},$((($end - $start) / 1000000))" >> ${log_file_name}  