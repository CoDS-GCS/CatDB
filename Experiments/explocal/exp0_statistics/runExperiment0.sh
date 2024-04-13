#!/bin/bash

dataset=$1


exp_path="$(pwd)"
data_path="${exp_path}/data"
statistics_path="${exp_path}/results/statistics"
mkdir -p ${statistics_path}

log_file_name="${exp_path}/results/statistics/dataset_overview.dat"
statistic_file_name="${exp_path}/results/statistics/${dataset}_statistics.dat"

# Run Data Profiling
data_profile_path="${data_path}/${dataset}/data_profile_full"
./explocal/exp0_statistics/runExperiment0_Data_Profile.sh ${dataset} ${data_profile_path}

# CatDB
./explocal/exp0_statistics/runExperiment0_CatDB.sh ${dataset} gpt-4-turbo Yes ${log_file_name} ${statistic_file_name} ${data_profile_path}
./explocal/exp0_statistics/runExperiment0_CatDB.sh ${dataset} gpt-4-turbo No ${log_file_name} ${statistic_file_name} ${data_profile_path}

# CAAFE
./explocal/exp0_statistics/runExperiment0_CAAFE.sh ${dataset} gpt-4-turbo Yes ${log_file_name}
./explocal/exp0_statistics/runExperiment0_CAAFE.sh ${dataset} gpt-4-turbo No ${log_file_name}