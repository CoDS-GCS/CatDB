#!/bin/bash

exp_path="$(pwd)"
root_patch="${exp_path}/catdb-results/cleaning"
CMDPatch=./explocal/exp1_cleaning/runExperiment1_Clean_Patch.sh 


dataset=$1
source_dataset_name=$2
patch_src=$3
split_dataset=$4

python_fname="gemini-1.5-pro-latest-CatDB-Data-Cleaning-iteration-1-part"

# $CMDPatch Midwest-Survey Midwest-Survey "${root_patch}/Midwest-Survey/${python_fname}-0-RUN.py" True
# $CMDPatch WiFi WiFi "${root_patch}/WiFi/${python_fname}-0-RUN.py" True
# $CMDPatch Utility Utility "${root_patch}/Utility/${python_fname}-0-RUN.py" True
# $CMDPatch EU-IT EU-IT "${root_patch}/EU-IT/${python_fname}-0-RUN.py" True
# $CMDPatch Etailing Etailing "${root_patch}/Etailing/${python_fname}-0-RUN.py" False
# $CMDPatch Etailing Etailing_clean "${root_patch}/Etailing/${python_fname}-1-RUN.py" True
