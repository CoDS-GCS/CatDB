#!/bin/bash

./load-spark-java11.sh

# setup, run experiments, plots
#./run1SetupDependencies.sh;
#./run2SetupBaseLines.sh;
./run3DownloadData.sh;
#./run4PrepareData.sh;
#./run5LocalExperiments.sh;
#./run6PlotResults.sh; 

#./explocal/runMergeResults.sh