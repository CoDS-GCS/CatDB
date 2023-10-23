#!/bin/bash

# Experiment1 Micro-Benchmark Identification Part

bcmd=./explocal/exp1_micor_bench/runExperiment1_Spark.sh

$bcmd "ca.concordia.ReadAndMergeByDF" "YELP" "N" ReaderCSVDataFrame Experiment1.dat
$bcmd "ca.concordia.ReadAndMergeBySQL" "YELP" "Q1" ReaderCSVDataFrame Experiment1.dat
#$bcmd "ca.concordia.ReadAndMergeBySQL" "YELP" "Q2" ReaderCSVDataFrame Experiment1.dat
#$bcmd "ca.concordia.ReadAndMergeBySQL" "YELP" "Q3" ReaderCSVDataFrame Experiment1.dat
#$bcmd "ca.concordia.ReadAndMergeBySQL" "YELP" "Q5" ReaderCSVDataFrame Experiment1.dat



