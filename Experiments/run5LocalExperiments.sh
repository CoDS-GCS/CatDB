#!/bin/bash

# clean original results
rm -rf results/*;
mkdir -p results;

exp_path="$(pwd)"

# Add headers to log files
echo "dataset,platform,time" >> "${exp_path}/results/Experiment1_Data_Profile.dat"
echo "dataset,platform,time,constraint" >> "${exp_path}/results/runExperiment1_AutoML.dat"

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