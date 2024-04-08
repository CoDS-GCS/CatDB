#!/bin/bash

dataset=$1

exp_path="$(pwd)"
data_path="${exp_path}/data"
data_profile_path="${data_path}/${dataset}/data_profile_full"

# Run Data Profiling
#./explocal/exp0_statistics/runExperiment0_Data_Profile.sh ${dataset} ${data_profile_path}

statistics_path="${exp_path}/results/statistics"
mkdir -p ${statistics_path}

# Run Statistics 
rm -rf "${statistics_path}/${dataset}" # clean-up
mkdir "${statistics_path}/${dataset}"

metadata_path="${data_path}/${dataset}/${dataset}.yaml"
output_path="${exp_path}/results/statistics/${dataset}"

cd "${exp_path}/setup/Baselines/CatDB/"
source venv/bin/activate

SCRIPT_base="python main_statistics.py --metadata-path ${metadata_path} \
        --data-profile-path ${data_profile_path} \
        --llm-model gpt-4 \
        --output-path ${output_path}"

SCRIPT_with_description="${SCRIPT_base} --dataset-description YES" 
SCRIPT_without_description="${SCRIPT_base} --dataset-description NO"        

echo ${SCRIPT_with_description}
$SCRIPT_with_description

echo ${SCRIPT_without_description}
$SCRIPT_without_description