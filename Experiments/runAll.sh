#!/bin/bash

mkdir -p setup

# setup, run experiments, plots
./run1SetupDependencies.sh;
./run2SetupBaseLines.sh;
./run3DownloadData.sh;
# ./run4GenerateData.sh;
./run5LocalExperiments.sh;
# ./run6PlotResults.sh; 