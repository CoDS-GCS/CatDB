#!/bin/bash


# This script runs all local experiments on the specified scale-up machine.

# clean original results
rm -rf results/*;
mkdir -p results;

echo "baseline,dataset,time_left,per_run_time_limit,time" > results/runExperiment1.dat
echo "baseline,accuracy,time_left,per_run_time_limit,max_models" > results/runExperiment1.dat.acu

for rp in {1..1}; do  
    #./explocal/exp1_systematic/runExperiment1.sh TelcoCustomerChurn Churn runExperiment1
    #./explocal/exp1_systematic/runExperiment1.sh kdd99 label runExperiment1
    ./explocal/exp1_systematic/runExperiment1.sh product_backorders went_on_backorder runExperiment1
    # ./explocal/exp1_systematic/runExperiment1.sh bank-additional-full y runExperiment1
    # ./explocal/exp1_systematic/runExperiment1.sh diabetes_prediction diabetes runExperiment1
    # ./explocal/exp1_systematic/runExperiment1.sh smoking_driking DRK_YN runExperiment1   

    #./explocal/exp1_systematic/runExperiment1.sh titanic Survived runExperiment1
     
    
done