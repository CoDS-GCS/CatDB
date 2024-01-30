#!/bin/bash

dataset=$1
task_type=$2

# Run CatDB Generated piplines
./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} SCHEMA NA Random 0 ${task_type} gpt-4 
./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} DISTINCT NA Random 0 ${task_type} gpt-4
./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} MISSING_VALUE NA Random 0 ${task_type} gpt-4
./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} NUMERIC_STATISTIC NA Random 0 ${task_type} gpt-4
./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} CATEGORICAL_VALUE NA Random 0 ${task_type} gpt-4
./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} DISTINCT_MISSING_VALUE NA Random 0 ${task_type} gpt-4
./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} MISSING_VALUE_NUMERIC_STATISTIC NA Random 0 ${task_type} gpt-4
./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} MISSING_VALUE_CATEGORICAL_VALUE NA Random 0 ${task_type} gpt-4
./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} NUMERIC_STATISTIC_CATEGORICAL_VALUE NA Random 0 ${task_type} gpt-4
./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} ALL NA Random 0 ${task_type} gpt-4

# Run AutoML Framework
#./explocal/exp2_micro_benchmark/runExperiment2_AutoML.sh ${dataset}
