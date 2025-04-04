#!/bin/bash

# clean original results
rm -rf results/*;
mkdir -p results;
mkdir -p catdb-results;

exp_path="$(pwd)"

# Add headers to log files
echo "dataset,task_type,platform,time" >> "${exp_path}/results/Experiment2_Data_Profile.dat"
echo "dataset,time" >> "${exp_path}/results/Experiment2_CSVDataReader.dat"
echo "dataset,source_data,time" >> "${exp_path}/results/Experiment1_Augmentation.dat"
echo "dataset,time" >> "${exp_path}/results/Experiment1_SAGA.dat"

cd ${exp_path}

# CMD=./explocal/exp0_statistics/runExperiment0.sh
# CMD=./explocal/exp1_cleaning/runExperiment1.sh
# CMDPatch=./explocal/exp1_cleaning/runPatch.sh
# CMD=./explocal/exp2_catdb/runExperiment2.sh
# CMD=./explocal/exp3_baselines/runExperiment3.sh

# $CMDPatch

# $CMD oml_dataset_2_rnc binary # Breast-w -> M1
# $CMD oml_dataset_4_rnc binary # Credit-g -> M1
# $CMD oml_dataset_5_rnc binary # Diabetes -> M1
# $CMD oml_dataset_33_rnc binary # Nomao -> M1
# $CMD oml_dataset_34_rnc multiclass # Gas-Drift -> M1
# $CMD oml_dataset_35_rnc multiclass # Volkert -> M1

# $CMD oml_dataset_20_rnc multiclass # Walking-Activity -> M2
# $CMD oml_dataset_6_rnc binary # Tic-Tac-Toe -> M2
# $CMD oml_dataset_3_rnc multiclass # CMC -> M2
# $CMD oml_dataset_22_rnc regression # Bike-Sharing -> M2

# $CMD oml_dataset_24_rnc regression # NYC -> M2
# $CMD oml_dataset_23_rnc regression # House-Sales -> M2
# $CMD Airline multiclass  #-> M2
# $CMD IMDB-IJS binary # -> M2
# $CMD Accidents multiclass # -> M2
# $CMD Financial multiclass # -> M2

#$CMD EU-IT multiclass # OK
#$CMD Etailing multiclass # OK
#$CMD Midwest-Survey multiclass # OK
#$CMD WiFi binary # OK
#$CMD Utility regression # OK
#$CMD Yelp multiclass

# $CMD KDD98 binary

# End to End Datasets
# $CMD Volkert-out-0.01-np-0-nc-0-mv-0 multiclass # Volkert
# $CMD Volkert-out-0.02-np-0-nc-0-mv-0 multiclass # Volkert
# $CMD Volkert-out-0.03-np-0-nc-0-mv-0 multiclass # Volkert
# $CMD Volkert-out-0.04-np-0-nc-0-mv-0 multiclass # Volkert
# $CMD Volkert-out-0.05-np-0-nc-0-mv-0 multiclass # Volkert

# $CMD Volkert-out-0-np-1-nc-180-mv-0.1 multiclass # Volkert
# $CMD Volkert-out-0-np-1-nc-180-mv-0.2 multiclass # Volkert
# $CMD Volkert-out-0-np-1-nc-180-mv-0.3 multiclass # Volkert
# $CMD Volkert-out-0-np-1-nc-180-mv-0.4 multiclass # Volkert
# $CMD Volkert-out-0-np-1-nc-180-mv-0.5 multiclass # Volkert

# $CMD Volkert-out-0.05-np-1-nc-180-mv-0.1 multiclass # Volkert
# $CMD Volkert-out-0.05-np-1-nc-180-mv-0.2 multiclass # Volkert
# $CMD Volkert-out-0.05-np-1-nc-180-mv-0.3 multiclass # Volkert
# $CMD Volkert-out-0.05-np-1-nc-180-mv-0.4 multiclass # Volkert
# $CMD Volkert-out-0.05-np-1-nc-180-mv-0.5 multiclass # Volkert

# $CMD Utility-out-0.01-np-0-nc-0-mv-0 regression # Utility
# $CMD Utility-out-0.02-np-0-nc-0-mv-0 regression # Utility
# $CMD Utility-out-0.03-np-0-nc-0-mv-0 regression # Utility
# $CMD Utility-out-0.04-np-0-nc-0-mv-0 regression # Utility
# $CMD Utility-out-0.05-np-0-nc-0-mv-0 regression # Utility

# $CMD Utility-out-0-np-1-nc-12-mv-0.1 regression # Utility
# $CMD Utility-out-0-np-1-nc-12-mv-0.2 regression # Utility
# $CMD Utility-out-0-np-1-nc-12-mv-0.3 regression # Utility
# $CMD Utility-out-0-np-1-nc-12-mv-0.4 regression # Utility
# $CMD Utility-out-0-np-1-nc-12-mv-0.5 regression # Utility

# $CMD Utility-out-0.05-np-1-nc-12-mv-0.1 regression # Utility
# $CMD Utility-out-0.05-np-1-nc-12-mv-0.2 regression # Utility
# $CMD Utility-out-0.05-np-1-nc-12-mv-0.3 regression # Utility
# $CMD Utility-out-0.05-np-1-nc-12-mv-0.4 regression # Utility
# $CMD Utility-out-0.05-np-1-nc-12-mv-0.5 regression # Utility
