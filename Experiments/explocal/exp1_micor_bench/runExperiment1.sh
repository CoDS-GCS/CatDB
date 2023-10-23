#!/bin/bash

# Experiment1 Micro-Benchmark Identification Part

bcmd=./explocal/exp1_micor_bench/runExperiment1_Spark.sh

# $bcmd "ca.concordia.ReadData" "YELP" "Q" ReadData Experiment1.dat

# $bcmd "ca.concordia.ReadAndMergeByDF" "YELP" "Q1" ReadAndMergeByDF Experiment1.dat
# $bcmd "ca.concordia.ReadAndMergeByDF" "YELP" "Q2" ReadAndMergeByDF Experiment1.dat
# $bcmd "ca.concordia.ReadAndMergeByDF" "YELP" "Q3" ReadAndMergeByDF Experiment1.dat

# $bcmd "ca.concordia.ReadAndMergeByDFCatalog" "YELP" "Q1" ReadAndMergeByDFCatalog Experiment1.dat
# $bcmd "ca.concordia.ReadAndMergeByDFCatalog" "YELP" "Q2" ReadAndMergeByDFCatalog Experiment1.dat
# $bcmd "ca.concordia.ReadAndMergeByDFCatalog" "YELP" "Q3" ReadAndMergeByDFCatalog Experiment1.dat



# $bcmd "ca.concordia.ReadAndMergeBySQL" "YELP" "Q1" ReadAndMergeBySQL Experiment1.dat
# $bcmd "ca.concordia.ReadAndMergeBySQL" "YELP" "Q2" ReadAndMergeBySQL Experiment1.dat
# $bcmd "ca.concordia.ReadAndMergeBySQL" "YELP" "Q3" ReadAndMergeBySQL Experiment1.dat

# $bcmd "ca.concordia.ReadAndMergeBySQLCatalog" "YELP" "Q1" ReadAndMergeBySQLCatalog Experiment1.dat
# $bcmd "ca.concordia.ReadAndMergeBySQLCatalog" "YELP" "Q2" ReadAndMergeBySQLCatalog Experiment1.dat
$bcmd "ca.concordia.ReadAndMergeBySQLCatalog" "YELP" "Q3" ReadAndMergeBySQLCatalog Experiment1.dat



