#!/bin/bash

exp_path="$(pwd)"
data_path="${exp_path}/data/SELA"
dataset=$1
llm_model=$2
task_type=$3
with_dataset_description=$4

metadata_path="${exp_path}/data/${dataset}/${dataset}.yaml"

# apply patch
patch_path="${exp_path}/setup/config/SELA"
des_path="${exp_path}/setup/Baselines/SELA/metagpt/ext/sela"
metagpt_path="${exp_path}/setup/Baselines/SELA/metagpt/"
cp -r "${patch_path}/dataset.py" "${des_path}/data"
cp -r "${patch_path}/main.py" "${des_path}/"
cp -r "${patch_path}/evaluation.py" "${des_path}/evaluation"
cp -r "${patch_path}/LogResults.py" "${des_path}/runner"
cp -r "${patch_path}/mcts.py" "${des_path}/runner"
cp -r "${patch_path}/google_gemini_api.py" "${metagpt_path}/provider/"
cp -r "${patch_path}/token_counter.py" "${metagpt_path}/utils/"
cp -r "${patch_path}/solution_designer.py" "${des_path}/insights/"
cp -r "${patch_path}/${llm_model}-config2.yaml" "${exp_path}/setup/Baselines/SELA/config/config2.yaml"
#

number_iteration=2
result_output_path="${exp_path}/sela-results/${dataset}"

mkdir -p "${exp_path}/sela-results"
rm -rf $result_output_path
mkdir -p "${result_output_path}"

cd "${exp_path}/setup/Baselines/SELA/"
source venv/bin/activate

## Prepare datasets for SELA baseline:
# sela_path="${exp_path}/setup/Baselines/SELA/"
# #sela_work_dir="${exp_path}/results/SELA/workspace"
# sela_work_dir="${result_output_path}/results/SELA/workspace"
# sela_role_dir="${exp_path}/results/SELA/storage"

# mkdir -p "${exp_path}/results/SELA"
# mkdir -p ${sela_data_out_path}
# mkdir -p ${sela_work_dir}
# mkdir -p ${sela_role_dir}

# cd "${exp_path}/setup/Baselines/SELA/metagpt/ext/sela"

# echo "datasets_dir: \"${data_path}\""  > data.yaml
# echo "work_dir: \"${sela_work_dir}\""  >> data.yaml
# echo "role_dir: \"${sela_role_dir}\""  >> data.yaml

# output_path="${exp_path}/results/Experiment1_LLM_SELA_ProcessDataset.dat"
# CMD="python data/dataset.py --root-data-path ${exp_path}/data \
#      --metadata-path ${metadata_path} \
#      --llm-model ${llm_model} \
#      --output-path ${output_path}"  
# $CMD

# Run SELA
output_path="${exp_path}/results/Experiment3_LLM_SELA.dat"

cd "${exp_path}/setup/Baselines/SELA/metagpt/ext/sela"
SCRIPT="python -Wignore main.py --exp_mode mcts \
        --task ${dataset} \
        --rollouts ${number_iteration}\
        --output-path ${result_output_path}  \
        --llm-model ${llm_model} \
        --result-output-path ${output_path} \
        --task-type ${task_type} \
        --role-timeout 1000"

echo ${SCRIPT}
$SCRIPT