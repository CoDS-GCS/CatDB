#!/bin/bash

# Experiment1 Micro-Benchmark Identification Part

bcmd=./explocal/exp1_micor_bench/runExperiment1_Spark.sh

$bcmd "ca.concordia.ReaderCSVBySQL" "YELP" ReaderCSVDataFrame Experiment1.dat

