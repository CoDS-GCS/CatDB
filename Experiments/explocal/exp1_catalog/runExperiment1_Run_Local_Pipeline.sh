#!/bin/bash

exp_path="$(pwd)"
data_path="${exp_path}/data"
dataset=$1
prompt_representation_type=$2
prompt_samples_type=$3
prompt_number_samples=$4
task_type=$5
llm_model=$6
with_dataset_description=$7
number_iteration=$8
dataset_train=$9
dataset_test=${10}

pipeline_path="${exp_path}/archive/VLDB2025/catdb-results/${dataset}"

result_output_path="${exp_path}/results/Experiment1_Local_Pipeline.dat"

cd "${exp_path}/setup/Baselines/CatDB/"
source venv/bin/activate

SCRIPT="python -Wignore main_runpipeline.py --root-data-path ${data_path} \
        --dataset-name ${dataset} \
        --task-type ${task_type} \
        --prompt-representation-type ${prompt_representation_type} \
        --prompt-samples-type ${prompt_samples_type} \
        --prompt-number-samples ${prompt_number_samples} \
        --prompt-number-iteration ${number_iteration} \
        --llm-model ${llm_model} \
        --pipeline-path ${pipeline_path} \
        --result-output-path ${result_output_path} \
        --dataset-description ${with_dataset_description} \
        --dataset-train ${dataset_train} \
        --dataset-test ${dataset_test}"


echo ${SCRIPT}

$SCRIPT