#!/bin/bash

exp_path="$(pwd)"
data_path="${exp_path}/data/SELA"
dataset=$1
llm_model=$2
classifier=$3
with_dataset_description=$4

# apply patch
patch_path="$(pwd)/setup/config/SELA"
des_path="$(pwd)/setup/Baselines/SELA/metagpt/ext/sela"
metagpt_path="$(pwd)/setup/Baselines/SELA/metagpt/"
cp -r "${patch_path}/DatasetPrepare.py" "${des_path}/data"
cp -r "${patch_path}/main.py" "${des_path}/"
cp -r "${patch_path}/evaluation.py" "${des_path}/evaluation"
cp -r "${patch_path}/LogResults.py" "${des_path}/runner"
cp -r "${patch_path}/mcts.py" "${des_path}/runner"
cp -r "${patch_path}/google_gemini_api.py" "${metagpt_path}/provider/"
# cp -r "${patch_path}/cost_manager.py" "${des_path}/utils/"
cp -r "${patch_path}/token_counter.py" "${des_path}/utils/"
#

metadata_path="${data_path}/${dataset}/${dataset}.yaml"
number_iteration=2
result_output_path="${exp_path}/sela-results/${dataset}"

date=$(date '+%Y-%m-%d-%H-%M-%S')
system_log="${exp_path}/system-log-${date}.dat"

mkdir -p "${exp_path}/sela-results"
mkdir -p "${result_output_path}"

output_path="${exp_path}/results/Experiment1_LLM_SELA.dat"

cd "${exp_path}/setup/Baselines/SELA/"
source venv/bin/activate

cd metagpt/ext/sela
SCRIPT="python -Wignore main.py --exp_mode mcts \
        --task ${dataset} \
        --rollouts ${number_iteration}\
        --output-path ${output_path} \
        --llm-model ${llm_model} \
        --result-output-path ${result_output_path} \
        --role-timeout 1000"

echo ${SCRIPT}
$SCRIPT