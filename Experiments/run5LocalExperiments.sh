#!/bin/bash

# clean original results
rm -rf results/*;
mkdir -p results;
mkdir -p catdb-results;

exp_path="$(pwd)"

# Add headers to log files

echo "dataset,platform,time,constraint" >> "${exp_path}/results/Experiment1_AutoML.dat"

./explocal/exp1_catalog.sh
