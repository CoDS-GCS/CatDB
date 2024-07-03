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

cd ${exp_path}

#CMD=./explocal/exp0_statistics/runExperiment0.sh
#CMD=./explocal/exp1_catalog/runExperiment1.sh
#CMD=./explocal/exp2_micro_benchmark/runExperiment2.sh 
CMD=./explocal/exp3_end_to_end/runExperiment3.sh 


# $CMD Balance-Scale multiclass # Balance-Scale
# $CMD Breast-w binary # Breast-w
# $CMD CMC multiclass # CMC
# $CMD Credit-g binary # Credit-g
# $CMD Diabetes binary # Diabetes
# $CMD Tic-Tac-Toe binary # Tic-Tac-Toe
# $CMD Eucalyptus multiclass # Eucalyptus
# $CMD PC1 binary # PC1
# $CMD Jungle-Chess multiclass # Jungle-Chess

# $CMD Higgs binary # Higgs
# $CMD Skin binary # Skin

# $CMD Traffic multiclass # Traffic
# $CMD Walking-Activity multiclass # Walking-Activity
# $CMD Black-Friday regression # Black-Friday
# $CMD Bike-Sharing regression # Bike-Sharing
# $CMD House-Sales regression # House-Sales
# $CMD NYC regression # NYC
# $CMD Airlines-DepDelay regression # Airlines-DepDelay


$CMD oml_dataset_1_rnc multiclass # Balance-Scale
$CMD oml_dataset_2_rnc binary # Breast-w
$CMD oml_dataset_3_rnc multiclass # CMC
$CMD oml_dataset_4_rnc binary # Credit-g
$CMD oml_dataset_5_rnc binary # Diabetes
$CMD oml_dataset_6_rnc binary # Tic-Tac-Toe
$CMD oml_dataset_7_rnc multiclass # Eucalyptus
$CMD oml_dataset_8_rnc binary # PC1
$CMD oml_dataset_10_rnc multiclass # Jungle-Chess

$CMD oml_dataset_11_rnc binary # Higgs
$CMD oml_dataset_12_rnc binary # Skin
$CMD oml_dataset_19_rnc multiclass # Traffic
$CMD oml_dataset_20_rnc multiclass # Walking-Activity

$CMD oml_dataset_21_rnc regression # Black-Friday
$CMD oml_dataset_22_rnc regression # Bike-Sharing
$CMD oml_dataset_23_rnc regression # House-Sales
$CMD oml_dataset_24_rnc regression # NYC
$CMD oml_dataset_25_rnc regression # Airlines-DepDelay

# $CMD oml_dataset_27_rnc multiclass # Yolanda
#$CMD oml_dataset_28_rnc binary # SF-Police-Incidents
# $CMD oml_dataset_29_rnc binary # Numerai28.6
# $CMD oml_dataset_30_rnc binary # Porto-Seguro
# $CMD oml_dataset_31_rnc multiclass # First-Order
# $CMD oml_dataset_32_rnc multiclass # Helena

######################################################
# $CMD Airlines binary # Airlines
# $CMD Click-Prediction binary # Click-Prediction
# $CMD Census-Augmented binary # Census-Augmented
# $CMD Heart-Statlog binary # Heart-Statlog
# $CMD KDDCup99 multiclass # KDDCup99
# $CMD Road-Safety multiclass # Road-Safety
# $CMD Drug-Directory multiclass # Drug-Directory
# $CMD Adult binary # Adult

# $CMD oml_dataset_26_rnc binary # Adult
# $CMD oml_dataset_14_rnc binary # Census-Augmented
# $CMD oml_dataset_15_rnc binary # Heart-Statlog
# $CMD oml_dataset_9_rnc binary # Airlines
# $CMD oml_dataset_13_rnc binary # Click-Prediction
# $CMD oml_dataset_16_rnc multiclass # KDDCup99
# $CMD oml_dataset_17_rnc multiclass # Road-Safety
# $CMD oml_dataset_18_rnc multiclass # Drug-Directory