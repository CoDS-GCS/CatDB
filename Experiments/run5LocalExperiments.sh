#!/bin/bash

# clean original results
rm -rf results/*;
mkdir -p results;

declare -a datasets=("dorothea" "christine" "jasmine" "philippine" "madeline" "sylvine" "albert" "evita" "digits" "newsgroups" "dilbert" "fabert" "robert" "volkert" "dionis" "jannis" "wallis" "helena" "cadata" "flora" "yolanda")

# Add headers to log files
echo "dataset,platform,time" >> results/Experiment1_Data_Profile.dat
echo "dataset,platform,time,constraint" >> results/runExperiment1_AutoML.dat

for dataset in "${datasets[@]}"; do
  ./explocal/exp1_systematic/runExperiment1.sh ${dataset}
done

