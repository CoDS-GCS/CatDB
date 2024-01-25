#!/bin/bash

exp_path="$(pwd)"
data_source_path="${exp_path}/data"
data_source_name=$1
prompt_representation_type=$2
prompt_example_type=$3
prompt_number_example=$4
task_type=$5
llm_model=$6
log_file_name="${exp_path}/results/Experiment2_CatDB_LLM_Pipe_Run.dat"

# Run LLM's generated code
pipeline_path="${exp_path}/catdb-results/${data_source_name}/${llm_model}"
python_script="${data_source_name}-${prompt_representation_type}-${prompt_example_type}-${prompt_number_example}-SHOT-${llm_model}"

file_path="${pipeline_path}/${python_script}.py"
if test -f "$file_path"; then

    cd ${pipeline_path}

    rm -rf venv
    rm -rf requirements.txt
    python -m venv venv
    source venv/bin/activate

    #Then install the dependencies:
    pip install pipreqs
    pipreqs --force --mode no-pin ${pipeline_path}
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

    cd ${exp_path}
    rm -rf "${pipeline_path}/${python_script}.log"

    SCRIPT="python ${file_path} > ${pipeline_path}/${python_script}.log"

    echo ${SCRIPT}

    start=$(date +%s%N)
    bash -c "${SCRIPT}"
    end=$(date +%s%N)

    echo "${data_source_name},${llm_model},${prompt_representation_type},${prompt_example_type},${prompt_number_example},${task_type},$((($end - $start) / 1000000))," >> $log_file_name
fi    

# clean-up
cd ${pipeline_path}
rm -rf venv