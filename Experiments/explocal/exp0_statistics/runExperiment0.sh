#!/bin/bash

dataset=$1

exp_path="$(pwd)"
data_path="${exp_path}/data"
statistics_path="${exp_path}/results/statistics"
mkdir -p ${statistics_path}

log_file_name="${exp_path}/results/statistics/dataset_overview.dat"
statistic_file_name="${exp_path}/results/statistics/${dataset}_statistics.dat"

data_profile_path="${exp_path}/catalog/${dataset}/data_profile"

cd "${exp_path}/setup/Baselines/CatDB/"
source venv/bin/activate

SCRIPT="python main_statistics.py \
        --dataset-name ${dataset} \
        --data-profile-path ${data_profile_path} \
        --log-file-name ${log_file_name} \
        --statistic-file-name ${statistic_file_name}"

echo ${SCRIPT}

$SCRIPT
