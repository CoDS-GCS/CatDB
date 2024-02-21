#!/bin/bash

dataset=$1
task_type=$2




# Run CatDB Generated piplines
CMD=./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh

$CMD ${dataset} SCHEMA Random 0 ${task_type} gpt-4 
$CMD ${dataset} DISTINCT Random 0 ${task_type} gpt-4
$CMD ${dataset} MISSING_VALUE Random 0 ${task_type} gpt-4
$CMD ${dataset} NUMERIC_STATISTIC Random 0 ${task_type} gpt-4
$CMD ${dataset} CATEGORICAL_VALUE Random 0 ${task_type} gpt-4
$CMD ${dataset} DISTINCT_MISSING_VALUE Random 0 ${task_type} gpt-4
$CMD ${dataset} MISSING_VALUE_NUMERIC_STATISTIC Random 0 ${task_type} gpt-4
$CMD ${dataset} MISSING_VALUE_CATEGORICAL_VALUE Random 0 ${task_type} gpt-4
$CMD ${dataset} NUMERIC_STATISTIC_CATEGORICAL_VALUE Random 0 ${task_type} gpt-4
$CMD ${dataset} ALL Random 0 ${task_type} gpt-4

# Run AutoML Framework
#./explocal/exp2_micro_benchmark/runExperiment2_AutoML.sh ${dataset}


