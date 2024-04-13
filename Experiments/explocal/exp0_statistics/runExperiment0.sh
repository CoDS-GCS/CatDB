#!/bin/bash

dataset=$1


exp_path="$(pwd)"
statistics_path="${exp_path}/results/statistics"
mkdir -p ${statistics_path}

log_file_name="${exp_path}/results/statistics/dataset_overview.dat"
statistic_file_name="${exp_path}/results/statistics/${dataset}_statistics.dat"

# CatDB
./explocal/exp0_statistics/runExperiment0_CatDB.sh ${dataset} gpt-4-turbo Yes ${log_file_name} ${statistic_file_name}
./explocal/exp0_statistics/runExperiment0_CatDB.sh ${dataset} gpt-4-turbo No ${log_file_name} ${statistic_file_name}

# CAAFE
./explocal/exp0_statistics/runExperiment0_CAAFE.sh ${dataset} gpt-4-turbo Yes ${log_file_name}
./explocal/exp0_statistics/runExperiment0_CAAFE.sh ${dataset} gpt-4-turbo No ${log_file_name}