#!/bin/bash

dataset=$1
task_type=$2
test=$3

exp_path="$(pwd)"

# Run Data Profiling
#./explocal/exp0_statistics/runExperiment0_Data_Profile.sh ${dataset}

statistics_path="${exp_path}/results/statistics"
mkdir -p ${statistics_path}

# clean-up
rm -rf "${statistics_path}/${dataset}"
mkdir "${statistics_path}/${dataset}"

# Run Statistics 
CMD=./explocal/exp0_statistics/runExperiment0_Statistics.sh

$CMD ${dataset} SCHEMA NA Random 0 ${task_type} gpt-4 $test
$CMD ${dataset} DISTINCT NA Random 0 ${task_type} gpt-4 $test
$CMD ${dataset} MISSING_VALUE NA Random 0 ${task_type} gpt-4 $test
$CMD ${dataset} NUMERIC_STATISTIC NA Random 0 ${task_type} gpt-4 $test
$CMD ${dataset} CATEGORICAL_VALUE NA Random 0 ${task_type} gpt-4 $test
$CMD ${dataset} DISTINCT_MISSING_VALUE NA Random 0 ${task_type} gpt-4 $test
$CMD ${dataset} MISSING_VALUE_NUMERIC_STATISTIC NA Random 0 ${task_type} gpt-4 $test
$CMD ${dataset} MISSING_VALUE_CATEGORICAL_VALUE NA Random 0 ${task_type} gpt-4 $test
$CMD ${dataset} NUMERIC_STATISTIC_CATEGORICAL_VALUE NA Random 0 ${task_type} gpt-4 $test
$CMD ${dataset} ALL NA Random 0 ${task_type} gpt-4 $test
