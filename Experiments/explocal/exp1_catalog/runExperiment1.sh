#!/bin/bash

dataset=$1
task_type=$2

exp_path="$(pwd)"

# Run Data Profiling Experiments
#./explocal/exp1_catalog/runExperiment1_Data_Profile.sh $dataset

# Rnn CSV Data Reader Experiment
#./explocal/exp1_catalog/runExperiment1_CSVDataReader.sh $dataset

# Run Prompt and LLM Pipeline Generation Experiments
#./explocal/exp1_systematic/runExperiment1_LLM_Pipe_Gen.sh ${dataset} TEXT Random 0 1 ${task_type} gpt-4
./explocal/exp1_catalog/runExperiment1_LLM_Pipe_Gen.sh ${dataset} TEXT Random 0 1 ${task_type} gpt-3.5-turbo
