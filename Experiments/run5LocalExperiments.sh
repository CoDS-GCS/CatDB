#!/bin/bash

# clean original results
rm -rf results/*;
mkdir -p results;
mkdir -p catdb-results;

exp_path="$(pwd)"

# Add headers to log files
echo "dataset,task_type,platform,time" >> "${exp_path}/results/Experiment1_Data_Profile.dat"
echo "dataset,time" >> "${exp_path}/results/Experiment1_CSVDataReader.dat"

cd ${exp_path}

# CMD=./explocal/exp0_statistics/runExperiment0.sh
# CMD=./explocal/exp1_cleaning/runExperiment1.sh
# CMDPatch=./explocal/exp1_cleaning/runPatch.sh
CMD=./explocal/exp1_catalog/runExperiment1.sh
# CMD=./explocal/exp2_micro_benchmark/runExperiment2.sh 
# CMD=./explocal/exp3_end_to_end/runExperiment3.sh
#CMD=./explocal/exp4_finetune/runExperiment4.sh
#CMD=./explocal/exp5_dataprepare/runExperiment5.sh

# $CMDPatch

$CMD oml_dataset_2_rnc binary # Breast-w
# $CMD oml_dataset_3_rnc multiclass # CMC
$CMD oml_dataset_4_rnc binary # Credit-g
$CMD oml_dataset_5_rnc binary # Diabetes
# $CMD oml_dataset_6_rnc binary # Tic-Tac-Toe
$CMD oml_dataset_33_rnc binary # Nomao
# $CMD oml_dataset_20_rnc multiclass # Walking-Activity
$CMD oml_dataset_34_rnc multiclass # Gas-Drift
$CMD oml_dataset_35_rnc multiclass # Volkert

#$CMD oml_dataset_22_rnc regression # Bike-Sharing
#$CMD oml_dataset_24_rnc regression # NYC
#$CMD oml_dataset_23_rnc regression # House-Sales


# $CMD Airline multiclass 
# $CMD IMDB-IJS binary 
# $CMD Accidents multiclass 
# $CMD Financial multiclass

# $CMD EU-IT multiclass # OK
# $CMD Etailing multiclass # OK
# $CMD Midwest-Survey multiclass # OK
# $CMD WiFi binary # OK
# $CMD Utility regression # OK
# $CMD Yelp multiclass 

#####################################################
# $CMD gen_dataset_54-out-0.05-np-1-nc-180-mv-0.1_rnc multiclass # Volkert
# $CMD gen_dataset_54-out-0.05-np-1-nc-180-mv-0.2_rnc multiclass # Volkert
# $CMD gen_dataset_54-out-0.05-np-1-nc-180-mv-0.3_rnc multiclass # Volkert
# $CMD gen_dataset_54-out-0.05-np-1-nc-180-mv-0.4_rnc multiclass # Volkert
# $CMD gen_dataset_54-out-0.05-np-1-nc-180-mv-0.5_rnc multiclass # Volkert

# $CMD gen_dataset_54-out-0.01-np-0-nc-0-mv-0_rnc multiclass # Volkert
# $CMD gen_dataset_54-out-0.02-np-0-nc-0-mv-0_rnc multiclass # Volkert
# $CMD gen_dataset_54-out-0.03-np-0-nc-0-mv-0_rnc multiclass # Volkert
# $CMD gen_dataset_54-out-0.05-np-0-nc-0-mv-0_rnc multiclass # Volkert
# $CMD gen_dataset_54-out-0.06-np-0-nc-0-mv-0_rnc multiclass # Volkert
# $CMD gen_dataset_54-out-0.07-np-0-nc-0-mv-0_rnc multiclass # Volkert
# $CMD gen_dataset_54-out-0.08-np-0-nc-0-mv-0_rnc multiclass # Volkert
# $CMD gen_dataset_54-out-0.09-np-0-nc-0-mv-0_rnc multiclass # Volkert
# $CMD gen_dataset_54-out-0.1-np-0-nc-0-mv-0_rnc multiclass # Volkert

# $CMD gen_dataset_54-out-0-np-1-nc-180-mv-0.1_rnc multiclass # Volkert
# $CMD gen_dataset_54-out-0-np-1-nc-180-mv-0.2_rnc multiclass # Volkert
# $CMD gen_dataset_54-out-0-np-1-nc-180-mv-0.3_rnc multiclass # Volkert
# $CMD gen_dataset_54-out-0-np-1-nc-180-mv-0.4_rnc multiclass # Volkert
# $CMD gen_dataset_54-out-0-np-1-nc-180-mv-0.5_rnc multiclass # Volkert


