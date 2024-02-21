#!/bin/bash

dataset=$1
task_type=$2

exp_path="$(pwd)"
data_path="${exp_path}/data"
data_profile_path="${data_path}/${dataset}/data_profile"


# Run Data Profiling Experiments
#./explocal/exp1_catalog/runExperiment1_Data_Profile.sh ${dataset} ${task_type}

# Rnn CSV Data Reader Experiment
#./explocal/exp1_catalog/runExperiment1_CSVDataReader.sh $dataset

# Run Prompt and LLM Pipeline Generation Experiments
CMD=./explocal/exp1_catalog/runExperiment1_LLM_Pipe_Gen.sh

$CMD ${dataset} ${data_profile_path} S Random 0 ${task_type} gpt-4 
$CMD ${dataset} ${data_profile_path} SDVC Random 0 ${task_type} gpt-4 
$CMD ${dataset} ${data_profile_path} SMVF Random 0 ${task_type} gpt-4 
$CMD ${dataset} ${data_profile_path} SSN Random 0 ${task_type} gpt-4 
$CMD ${dataset} ${data_profile_path} SCV Random 0 ${task_type} gpt-4 
$CMD ${dataset} ${data_profile_path} SDVCMVF Random 0 ${task_type} gpt-4 
$CMD ${dataset} ${data_profile_path} SDVCSN Random 0 ${task_type} gpt-4 
$CMD ${dataset} ${data_profile_path} SMVFSN Random 0 ${task_type} gpt-4 
$CMD ${dataset} ${data_profile_path} SMVFCV Random 0 ${task_type} gpt-4 
$CMD ${dataset} ${data_profile_path} SSNCV Random 0 ${task_type} gpt-4 
$CMD ${dataset} ${data_profile_path} ALL Random 0 ${task_type} gpt-4 


# PROMPT_FUNC = {"S": SchemaPrompt,
#                "SDVC": SchemaDistinctValuePrompt,
#                "SMVF": SchemaMissingValueFrequencyPrompt,
#                "SSN": SchemaStatisticNumericPrompt,
#                "SCV": SchemaCategoricalValuesPrompt,
#                "SDVCMVF": SchemaDistinctValueCountMissingValueFrequencyPrompt,
#                "SDVCSN": SchemaDistinctValueCountStatisticNumericPrompt,
#                "SMVFSN": SchemaMissingValueFrequencyStatisticNumericPrompt,
#                "SMVFCV": SchemaMissingValueFrequencyCategoricalValuesPrompt,
#                "SSNCV": SchemaStatisticNumericCategoricalValuesPrompt,
#                "ALL": AllPrompt
#                }
