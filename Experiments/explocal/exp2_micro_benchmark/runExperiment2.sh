#!/bin/bash

dataset=$1
task_type=$2

# Run CatDB Generated piplines

# SCHEMA
########
#./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} SCHEMA Random 0 1 ${task_type} gpt-4 
#./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} SCHEMA Random 0 1 ${task_type} gpt-3.5-turbo

# SCHEMA_STATISTIC
#./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} SCHEMA_STATISTIC Random 0 1 ${task_type} gpt-4 
#./explocal/exp2_micro_benchmark/runExperiment2_CatDB_LLM_Pipe_Run.sh ${dataset} SCHEMA_STATISTIC Random 0 1 ${task_type} gpt-3.5-turbo

# Run AutoML Framework
#./explocal/exp2_micro_benchmark/runExperiment2_AutoML.sh B #${dataset}
