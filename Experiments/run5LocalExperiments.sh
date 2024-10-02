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

#CMD=./explocal/exp0_statistics/runExperiment0.sh
CMD=./explocal/exp1_cleaning/runExperiment1.sh
#CMDPath=./explocal/exp1_cleaning/runPatch.sh
# CMD=./explocal/exp1_catalog/runExperiment1.sh
#CMD=./explocal/exp2_micro_benchmark/runExperiment2.sh 
#CMD=./explocal/exp3_end_to_end/runExperiment3.sh
#CMD=./explocal/exp4_finetune/runExperiment4.sh
#CMD=./explocal/exp5_dataprepare/runExperiment5.sh

#$CMDPath

# $CMD oml_dataset_1_rnc multiclass # Balance-Scale
# $CMD oml_dataset_2_rnc binary # Breast-w
# $CMD oml_dataset_3_rnc multiclass # CMC
# $CMD oml_dataset_4_rnc binary # Credit-g
# $CMD oml_dataset_5_rnc binary # Diabetes
# $CMD oml_dataset_6_rnc binary # Tic-Tac-Toe
# $CMD oml_dataset_10_rnc multiclass # Jungle-Chess

# $CMD oml_dataset_11_rnc binary # Higgs
# $CMD oml_dataset_12_rnc binary # Skin
# $CMD oml_dataset_33_rnc binary # Nomao

# $CMD oml_dataset_19_rnc multiclass # Traffic
# $CMD oml_dataset_20_rnc multiclass # Walking-Activity
# $CMD oml_dataset_34_rnc multiclass # Gas-Drift
# $CMD oml_dataset_35_rnc multiclass # Volkert

# $CMD oml_dataset_21_rnc regression # Black-Friday
# $CMD oml_dataset_22_rnc regression # Bike-Sharing
# $CMD oml_dataset_23_rnc regression # House-Sales
# $CMD oml_dataset_24_rnc regression # NYC


# $CMD Accidents multiclass 
# $CMD Airline multiclass 
# $CMD Financial multiclass 
# $CMD IMDB-IJS binary 
# $CMD Yelp regression 

# $CMD Midwest-Survey multiclass # OK
#$CMD WiFi binary # OK
# $CMD Utility regression # OK
# $CMD EU-IT multiclass # OK
$CMD Etailing multiclass # OK

# $CMD Relocated-Vehicles multiclass # OK
# $CMD Violations multiclass
# $CMD Health-Sciences multiclass
# $CMD Mid-Feed multiclass

# $CMD Lahman-2014 regression 
# $CMD Walmart regression 
# #$CMD Walmart-2014 regression 
# $CMD IMDB binary 
#$CMD Halloween multiclass
#$CMD Salaries multiclass
#$CMD US-Labor multiclass
#$CMD San-Francisco multiclass
#$CMD TSM-Habitat multiclass


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


