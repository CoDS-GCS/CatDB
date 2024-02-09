#!/bin/bash

dataset=$1
task_type=$2
test=$3

exp_path="$(pwd)"

# Run Data Profiling Experiments
#./explocal/exp1_catalog/runExperiment1_Data_Profile.sh ${dataset} ${task_type}

# Rnn CSV Data Reader Experiment
#./explocal/exp1_catalog/runExperiment1_CSVDataReader.sh $dataset

# Run Prompt and LLM Pipeline Generation Experiments
CMD=./explocal/exp1_catalog/runExperiment1_LLM_Pipe_Gen.sh

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
