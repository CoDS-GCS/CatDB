#!/bin/bash

# clean original results
rm -rf results/*;
mkdir -p results;
mkdir -p catdb-results;

exp_path="$(pwd)"

# Add headers to log files
echo "dataset,task_type,platform,time" >> "${exp_path}/results/Experiment1_Data_Profile.dat"
echo "dataset,time" >> "${exp_path}/results/Experiment1_CSVDataReader.dat"

echo "dataset,llm_model,prompt_representation_type,prompt_example_type,prompt_number_example,task_type,time,result" >> "${exp_path}/results/Experiment2_CatDB_LLM_Pipe_Run.dat"
# echo "dataset,platform,time,constraint" >> "${exp_path}/results/Experiment2_AutoML_Corresponding.dat"

# echo "dataset,platform,time,constraint" >> "${exp_path}/results/Experiment3_AutoML_1H.dat"
# echo "dataset,source,time" >> "${exp_path}/results/Experiment3_Hand-craft.dat"

cd ${exp_path}

#CMD=./explocal/exp0_statistics/runExperiment0.sh
CMD=./explocal/exp1_catalog/runExperiment1.sh
#CMD=./explocal/exp2_micro_benchmark/runExperiment2.sh 
#CMD=./explocal/exp3_end_to_end/runExperiment3.sh 


# Large Datasets
$CMD Higgs binary
$CMD Albert binary
$CMD Click-Prediction binary
$CMD Census-Augmented binary
$CMD Heart-Statlog binary
$CMD KDDCup99 multiclass
$CMD Road-Safety multiclass
$CMD Drug-Directory multiclass
$CMD Okcupid-Stem multiclass
$CMD Walking-Activity multiclass
$CMD PASS regression
$CMD Aloi regression
$CMD MD-MIX-Mini regression
$CMD Dionis regression
$CMD Meta-Album-BRD regression

# Small Datasets
$CMD Balance-Scale multiclass
$CMD Breast-w binary
$CMD CMC multiclass
$CMD Credit-g binary
$CMD Diabetes binary
$CMD Tic-Tac-Toe binary
$CMD Eucalyptus multiclass
$CMD PC1 binary
$CMD Airlines binary
$CMD Jungle-Chess multiclass


