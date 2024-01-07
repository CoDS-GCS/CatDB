#!/bin/bash

dataset=$1
task_type=$2
constraint=$3

exp_path="$(pwd)"

# Run Data Profiling Experiments
#log_file_name="${exp_path}/results/Experiment1_Data_Profile.dat"
#./explocal/exp1_systematic/runExperiment1_Data_Profile.sh $dataset $log_file_name

# Run CatDB Experiments
#log_file_name="${exp_path}/results/Experiment1_CatDB"
#./explocal/exp1_systematic/runExperiment1_CatDB.sh ${dataset} TEXT Random 0 1 ${task_type} gpt-4 ${log_file_name}
#./explocal/exp1_systematic/runExperiment1_CatDB.sh ${dataset} TEXT Random 0 1 ${task_type} gpt-3.5-turbo ${log_file_name}

# Run AutoML Baseline
log_file_name="${exp_path}/results/runExperiment1_AutoML.dat"
./explocal/exp1_systematic/runExperiment1_AutoML.sh $dataset $constraint $log_file_name 