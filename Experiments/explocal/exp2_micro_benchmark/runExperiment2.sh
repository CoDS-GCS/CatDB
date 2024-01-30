#!/bin/bash

dataset=$1
task_type=$2

# Run CatDB Generated piplines
./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} SCHEMA Random 0 ${task_type} gpt-4 
./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} DISTINCT Random 0 ${task_type} gpt-4
./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} MISSING_VALUE Random 0 ${task_type} gpt-4
./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} NUMERIC_STATISTIC Random 0 ${task_type} gpt-4
./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} CATEGORICAL_VALUE Random 0 ${task_type} gpt-4
./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} DISTINCT_MISSING_VALUE Random 0 ${task_type} gpt-4
./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} MISSING_VALUE_NUMERIC_STATISTIC Random 0 ${task_type} gpt-4
./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} MISSING_VALUE_CATEGORICAL_VALUE Random 0 ${task_type} gpt-4
./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} NUMERIC_STATISTIC_CATEGORICAL_VALUE Random 0 ${task_type} gpt-4
./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} ALL Random 0 ${task_type} gpt-4

# Run AutoML Framework
#./explocal/exp2_micro_benchmark/runExperiment2_AutoML.sh ${dataset}
