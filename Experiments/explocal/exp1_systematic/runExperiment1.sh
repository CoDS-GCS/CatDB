#!/bin/bash

dataset=$1
task_type=$2

# Run Data Profiling Experiments
#log_file_name=results/Experiment1_Data_Profile.dat
#./explocal/exp1_systematic/runExperiment1_Data_Profile.sh $dataset $log_file_name

# Run CatDB Experiments

# Run AutoML Baseline
log_file_name=results/runExperiment1_AutoML.dat
./explocal/exp1_systematic/runExperiment1_AutoML.sh $dataset $log_file_name