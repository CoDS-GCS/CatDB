#!/bin/bash

# clean original results
rm -rf results/*;
mkdir -p results;

exp_path="$(pwd)"

# Add headers to log files
echo "dataset,platform,time" >> "${exp_path}/results/Experiment1_Data_Profile.dat"
echo "dataset,platform,time,constraint" >> "${exp_path}/results/runExperiment1_AutoML.dat"

./explocal/exp1_systematic/runExperiment1.sh airlines Binary 
./explocal/exp1_systematic/runExperiment1.sh albert Binary 
./explocal/exp1_systematic/runExperiment1.sh covertype Multiclass 
./explocal/exp1_systematic/runExperiment1.sh dionis Multiclass 
./explocal/exp1_systematic/runExperiment1.sh adult Binary 
./explocal/exp1_systematic/runExperiment1.sh Amazon_employee_access Binary 
./explocal/exp1_systematic/runExperiment1.sh APSFailure Binary 
./explocal/exp1_systematic/runExperiment1.sh bank-marketing Binary 
./explocal/exp1_systematic/runExperiment1.sh connect-4 Multiclass 
./explocal/exp1_systematic/runExperiment1.sh Fashion-MNIST Multiclass 
./explocal/exp1_systematic/runExperiment1.sh guillermo Binary 
./explocal/exp1_systematic/runExperiment1.sh helena Multiclass 
./explocal/exp1_systematic/runExperiment1.sh higgs Binary 
./explocal/exp1_systematic/runExperiment1.sh jannis Multiclass 
./explocal/exp1_systematic/runExperiment1.sh jungle_chess_2pcs_raw_endgame_complete Multiclass 
./explocal/exp1_systematic/runExperiment1.sh KDDCup09_appetency Binary 
./explocal/exp1_systematic/runExperiment1.sh MiniBooNE Binary 
./explocal/exp1_systematic/runExperiment1.sh nomao Binary 
./explocal/exp1_systematic/runExperiment1.sh numerai28.6 Binary 
./explocal/exp1_systematic/runExperiment1.sh riccardo Binary 
./explocal/exp1_systematic/runExperiment1.sh robert Multiclass 
./explocal/exp1_systematic/runExperiment1.sh shuttle Multiclass 
./explocal/exp1_systematic/runExperiment1.sh volkert Multiclass 
./explocal/exp1_systematic/runExperiment1.sh Australian Binary 
./explocal/exp1_systematic/runExperiment1.sh blood-transfusion-service-center Binary 
./explocal/exp1_systematic/runExperiment1.sh car Multiclass 
./explocal/exp1_systematic/runExperiment1.sh christine Binary 
./explocal/exp1_systematic/runExperiment1.sh cnae-9 Multiclass 
./explocal/exp1_systematic/runExperiment1.sh credit-g Binary 
./explocal/exp1_systematic/runExperiment1.sh dilbert Multiclass 
./explocal/exp1_systematic/runExperiment1.sh fabert Multiclass 
./explocal/exp1_systematic/runExperiment1.sh jasmine Binary 
./explocal/exp1_systematic/runExperiment1.sh kc1 Binary 
./explocal/exp1_systematic/runExperiment1.sh kr-vs-kp Binary 
./explocal/exp1_systematic/runExperiment1.sh mfeat-factors Multiclass 
./explocal/exp1_systematic/runExperiment1.sh phoneme Binary 
./explocal/exp1_systematic/runExperiment1.sh segment Multiclass 
./explocal/exp1_systematic/runExperiment1.sh sylvine Binary 
./explocal/exp1_systematic/runExperiment1.sh vehicle Multiclass 

#Binary classification datasets:
###############################
# ./explocal/exp1_systematic/runExperiment1.sh dorothea BinaryClassification
# ./explocal/exp1_systematic/runExperiment1.sh christine BinaryClassification
# ./explocal/exp1_systematic/runExperiment1.sh jasmine BinaryClassification
# ./explocal/exp1_systematic/runExperiment1.sh philippine BinaryClassification
# ./explocal/exp1_systematic/runExperiment1.sh madeline BinaryClassification
#./explocal/exp1_systematic/runExperiment1.sh sylvine BinaryClassification
#./explocal/exp1_systematic/runExperiment1.sh albert BinaryClassification
#./explocal/exp1_systematic/runExperiment1.sh evita BinaryClassification

# # Multiclass classification datasets:
# ####################################
# ./explocal/exp1_systematic/runExperiment1.sh digits MulticlassClassification
# ./explocal/exp1_systematic/runExperiment1.sh newsgroups MulticlassClassification
# ./explocal/exp1_systematic/runExperiment1.sh dilbert MulticlassClassification
# ./explocal/exp1_systematic/runExperiment1.sh fabert MulticlassClassification
# ./explocal/exp1_systematic/runExperiment1.sh robert MulticlassClassification
# ./explocal/exp1_systematic/runExperiment1.sh volkert MulticlassClassification
# ./explocal/exp1_systematic/runExperiment1.sh dionis MulticlassClassification
# ./explocal/exp1_systematic/runExperiment1.sh jannis MulticlassClassification
# ./explocal/exp1_systematic/runExperiment1.sh wallis MulticlassClassification
# ./explocal/exp1_systematic/runExperiment1.sh helena MulticlassClassification

# # Regression datasets:
# ######################
# ./explocal/exp1_systematic/runExperiment1.sh cadata Regression
# ./explocal/exp1_systematic/runExperiment1.sh flora Regression
# ./explocal/exp1_systematic/runExperiment1.sh yolanda Regression