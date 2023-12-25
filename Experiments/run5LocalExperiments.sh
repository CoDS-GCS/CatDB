#!/bin/bash


# This script runs all local experiments on the specified scale-up machine.

# clean original results
rm -rf results/*;
mkdir -p results;

./explocal/exp1_systematic/runExperiment1.sh