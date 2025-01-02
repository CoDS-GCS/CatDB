#!/bin/bash

dataset=$1
task_type=$2

# Run AutoML Baseline
iteration=1
deafult_max_run_time=1
CMDAutoML=./explocal/exp3_baselines/runExperiment3_AutoML.sh
CMDAutoMLSAGA=./explocal/exp3_baselines/runExperiment3_AutoML_SAGA.sh
CMDCAAFE=./explocal/exp3_baselines/runExperiment3_LLM_CAAFE.sh
CMDAIDE=./explocal/exp3_baselines/runExperiment3_LLM_AIDE.sh

## AutoML Baselines
##------------------
# $CMDAutoML $dataset AutoSklearn ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest
# $CMDAutoML $dataset H2O  ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest
# $CMDAutoML $dataset Flaml  ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest
# $CMDAutoML $dataset Autogluon ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest

# $CMDAutoML $dataset AutoSklearn ${deafult_max_run_time} ${iteration} llama-3.1-70b-versatile
# $CMDAutoML $dataset H2O  ${deafult_max_run_time} ${iteration} llama-3.1-70b-versatile
# $CMDAutoML $dataset Flaml  ${deafult_max_run_time} ${iteration} llama-3.1-70b-versatile
# $CMDAutoML $dataset Autogluon ${deafult_max_run_time} ${iteration} llama-3.1-70b-versatile

# $CMDAutoML $dataset AutoSklearn ${deafult_max_run_time} ${iteration} gpt-4o
# $CMDAutoML $dataset H2O  ${deafult_max_run_time} ${iteration} gpt-4o
# $CMDAutoML $dataset Flaml  ${deafult_max_run_time} ${iteration} gpt-4o
# $CMDAutoML $dataset Autogluon ${deafult_max_run_time} ${iteration} gpt-4o

## AutoML + SAGA
## ------------
# $CMDAutoMLSAGA $dataset AutoSklearn ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest
# $CMDAutoMLSAGA $dataset H2O  ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest
# $CMDAutoMLSAGA $dataset Flaml  ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest
$CMDAutoMLSAGA $dataset Autogluon ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest


# $CMDAutoMLSAGA $dataset AutoSklearn ${deafult_max_run_time} ${iteration} llama-3.1-70b-versatile
# $CMDAutoMLSAGA $dataset H2O  ${deafult_max_run_time} ${iteration} llama-3.1-70b-versatile
# $CMDAutoMLSAGA $dataset Flaml  ${deafult_max_run_time} ${iteration} llama-3.1-70b-versatile
# $CMDAutoMLSAGA $dataset Autogluon ${deafult_max_run_time} ${iteration} llama-3.1-70b-versatile


# $CMDAutoMLSAGA $dataset AutoSklearn ${deafult_max_run_time} ${iteration} gpt-4o
# $CMDAutoMLSAGA $dataset H2O  ${deafult_max_run_time} ${iteration} gpt-4o
# $CMDAutoMLSAGA $dataset Flaml  ${deafult_max_run_time} ${iteration} gpt-4o
# $CMDAutoMLSAGA $dataset Autogluon ${deafult_max_run_time} ${iteration} gpt-4o


## AutoGen Baseline
#------------------
#$CMDAutoML $dataset AutoGen ${deafult_max_run_time} ${iteration} gpt-4o 
#$CMDAutoML $dataset AutoGen ${deafult_max_run_time} ${iteration} gemini-1.5-pro-latest 
#$CMDAutoML $dataset AutoGen ${deafult_max_run_time} ${iteration} llama-3.1-70b-versatile 


## CAAFE Baseline
## --------------
# $CMDCAAFE ${dataset} gemini-1.5-pro-latest TabPFN No
# $CMDCAAFE ${dataset} gemini-1.5-pro-latest RandomForest No

# $CMDCAAFE ${dataset} llama3-70b-8192 TabPFN No
# $CMDCAAFE ${dataset} llama3-70b-8192 RandomForest No

# $CMDCAAFE ${dataset} llama-3.1-70b-versatile TabPFN No
# $CMDCAAFE ${dataset} llama-3.1-70b-versatile RandomForest No

# $CMDCAAFE ${dataset} gpt-4o TabPFN No
# $CMDCAAFE ${dataset} gpt-4o RandomForest No


## AIDE Baseline
## --------------
# $CMDAIDE ${dataset} gemini-1.5-pro-latest 1
# $CMDAIDE ${dataset} gemini-1.5-pro-latest 2
# $CMDAIDE ${dataset} gemini-1.5-pro-latest 3
# $CMDAIDE ${dataset} gemini-1.5-pro-latest 4
# $CMDAIDE ${dataset} gemini-1.5-pro-latest 5
# $CMDAIDE ${dataset} gemini-1.5-pro-latest 6
# $CMDAIDE ${dataset} gemini-1.5-pro-latest 7
# $CMDAIDE ${dataset} gemini-1.5-pro-latest 8
# $CMDAIDE ${dataset} gemini-1.5-pro-latest 9
# $CMDAIDE ${dataset} gemini-1.5-pro-latest 10