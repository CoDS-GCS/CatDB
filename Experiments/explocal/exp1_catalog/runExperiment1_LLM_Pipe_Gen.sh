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
log_file_name="${exp_path}/results/Experiment1_LLM_Pipe_Gen.dat"

output_path="${exp_path}/catdb-results/${data_source_name}"
mkdir -p ${output_path}

output_path="${exp_path}/catdb-results/${data_source_name}/${llm_model}"
mkdir -p ${output_path}


cd "${exp_path}/setup/Baselines/CatDB/"
source venv/bin/activate

SCRIPT="python main.py --data-source-path ${data_source_path} \
        --data-source-name ${data_source_name} \
        --prompt-representation-type ${prompt_representation_type} \
        --suggested-model ${suggested_model} \
        --prompt-example-type ${prompt_example_type} \
        --prompt-number-example ${prompt_number_example} \
        --llm-model ${llm_model} \
        --output-path ${output_path}"

# sudo echo 3 >/proc/sys/vm/drop_caches && sudo sync
# sleep 3

echo ${SCRIPT}

start=$(date +%s%N)
$SCRIPT
end=$(date +%s%N)

echo "${data_source_name},${llm_model},${prompt_representation_type},${prompt_example_type},${prompt_number_example},${task_type},$((($end - $start) / 1000000))" >> ${log_file_name}      


if [[ "$test" == "test" ]]; then
    for itr in {1..10}; do 
        cd ${exp_path}
        ./explocal/exp1_catalog/runExperiment1_CatDB_LLM_Pipe_Test.sh $data_source_name $prompt_representation_type $prompt_example_type $prompt_number_example $task_type $llm_model $itr
    done
fi