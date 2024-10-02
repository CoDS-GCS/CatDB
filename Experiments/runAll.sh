#!/bin/bash

./load-spark-java11.sh

# setup, run experiments, plots
#./run1SetupDependencies.sh;
./run2SetupBaseLines.sh;
#./run3DownloadData.sh;
#./run4PrepareData.sh;
./run5LocalExperiments.sh;
#./run6PlotResults.sh; 

#./explocal/runMergeResults.sh


# Note: 
# HfXIyqtxFE.json
#  "table_name": "Accidents.csv",
#     "table_id": "Accidents/Accidents.csvAccidents.csv",
#     "column_name": "varnostni_pas_ali_celada",

# 3SICD2lh6m.json
# "column_name": "oseba.ime_upravna_enota",

#5Tiu5oRwYw.json


# Yelp:
# 9VRGappwBj.json >> address
# Zz0VA9poum.json >> category -> list