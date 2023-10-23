#!/bin/bash

export LOG4JPROP='explocal/log4j.properties'
export CMD="java -Xms28g -Xmx28g -Dlog4j.configuration=file:$LOG4JPROP"

mkdir -p setup

source load-had3.3-java11.sh

# setup, run experiments, plots
#./run1SetupDependencies.sh;
./run2SetupBaseLines.sh;
#./run3DownloadData.sh;
#./run4GenerateData.sh;
./run5LocalExperiments.sh;
#./run6PlotResults.sh; 



#https://aniket02.github.io/Yelp_Reviews/
#https://medium.com/@a0981639183/use-a-multiple-linear-regression-model-to-investigate-what-factors-most-affect-a-restaurants-yelp-d00b8dd264f0