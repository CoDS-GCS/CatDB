#!/bin/bash

exp_path="$(pwd)"
data_source_name=$1
prompt_representation_type=$2
prompt_example_type=$3
prompt_number_example=$4
task_type=$5
llm_model=$6
itr=$7
log_file_name="${exp_path}/results/Experiment1_LLM_Pipe_Test.dat"

cd ${exp_path}
pipeline_path="${exp_path}/catdb-results/${data_source_name}/${llm_model}"
python_script="${data_source_name}-${prompt_representation_type}-${prompt_example_type}-${prompt_number_example}-SHOT-${llm_model}"

file_path="${pipeline_path}/${python_script}.py"
error_path="${pipeline_path}/${python_script}.error"

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
    rm -rf $error_path

    SCRIPT="python -Wignore ${file_path} > ${pipeline_path}/${python_script}.log 2>${error_path} < /dev/null"
    bash -c "nohup ${SCRIPT}" 

    if test -f "$error_path"; then
        nrows=$(sed -n '$=' $error_path)
        if [ "$nrows" -gt "0" ]; then
            cp ${file_path} "${pipeline_path}/${python_script}.py.iteration_${itr}"
            cp ${error_path} "${error_path}.iteration_${itr}"

            cd "${exp_path}/setup/Baselines/CatDB/"
            source venv/bin/activate
            SCRIPT="python main_fix.py --pipeline-in-path ${file_path} \
                                       --pipeline-out-path ${file_path} \
                                       --prompt-out-path ${file_path}.${itr}.txt \
                                       --error-message-path ${error_path} \
                                       --llm-model ${llm_model}"
            echo $SCRIPT
            start=$(date +%s%N)
            $SCRIPT
            end=$(date +%s%N)
            echo "${data_source_name},${itr},${llm_model},${prompt_representation_type},${prompt_example_type},${prompt_number_example},${task_type},$((($end - $start) / 1000000))" >> ${log_file_name}                            
        else
            break
        fi
    else
        break
    fi
fi    

# clean-up
cd ${pipeline_path}
rm -rf venv  