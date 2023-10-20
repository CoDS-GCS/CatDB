#!/bin/bash

# This script runs all local experiments on the specified scale-up machine.

# clean original results
rm -rf results/*;
mkdir -p results;

for rp in {1..1}; do  
    ./explocal/exp1_micor_bench/runExperiment1.sh     
done