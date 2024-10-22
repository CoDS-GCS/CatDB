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
# $CMDPatch Yelp Yelp "${root_patch}/Yelp/${python_fname}-0-RUN.py" True

$CMDPatch Utility-out-0.01-np-0-nc-0-mv-0 Utility-out-0.01-np-0-nc-0-mv-0 "${root_patch}/Utility/${python_fname}-0-RUN.py" True
$CMDPatch Utility-out-0.02-np-0-nc-0-mv-0 Utility-out-0.02-np-0-nc-0-mv-0 "${root_patch}/Utility/${python_fname}-0-RUN.py" True
$CMDPatch Utility-out-0.03-np-0-nc-0-mv-0 Utility-out-0.03-np-0-nc-0-mv-0 "${root_patch}/Utility/${python_fname}-0-RUN.py" True
$CMDPatch Utility-out-0.04-np-0-nc-0-mv-0 Utility-out-0.04-np-0-nc-0-mv-0 "${root_patch}/Utility/${python_fname}-0-RUN.py" True
$CMDPatch Utility-out-0.05-np-0-nc-0-mv-0 Utility-out-0.05-np-0-nc-0-mv-0 "${root_patch}/Utility/${python_fname}-0-RUN.py" True
$CMDPatch Utility-out-0-np-1-nc-12-mv-0.1 Utility-out-0-np-1-nc-12-mv-0.1 "${root_patch}/Utility/${python_fname}-0-RUN.py" True
$CMDPatch Utility-out-0-np-1-nc-12-mv-0.2 Utility-out-0-np-1-nc-12-mv-0.2 "${root_patch}/Utility/${python_fname}-0-RUN.py" True
$CMDPatch Utility-out-0-np-1-nc-12-mv-0.3 Utility-out-0-np-1-nc-12-mv-0.3 "${root_patch}/Utility/${python_fname}-0-RUN.py" True
$CMDPatch Utility-out-0-np-1-nc-12-mv-0.4 Utility-out-0-np-1-nc-12-mv-0.4 "${root_patch}/Utility/${python_fname}-0-RUN.py" True
$CMDPatch Utility-out-0-np-1-nc-12-mv-0.5 Utility-out-0-np-1-nc-12-mv-0.5 "${root_patch}/Utility/${python_fname}-0-RUN.py" True
$CMDPatch Utility-out-0.05-np-1-nc-12-mv-0.1 Utility-out-0.05-np-1-nc-12-mv-0.1 "${root_patch}/Utility/${python_fname}-0-RUN.py" True
$CMDPatch Utility-out-0.05-np-1-nc-12-mv-0.2 Utility-out-0.05-np-1-nc-12-mv-0.2 "${root_patch}/Utility/${python_fname}-0-RUN.py" True
$CMDPatch Utility-out-0.05-np-1-nc-12-mv-0.3 Utility-out-0.05-np-1-nc-12-mv-0.3 "${root_patch}/Utility/${python_fname}-0-RUN.py" True
$CMDPatch Utility-out-0.05-np-1-nc-12-mv-0.4 Utility-out-0.05-np-1-nc-12-mv-0.4 "${root_patch}/Utility/${python_fname}-0-RUN.py" True
$CMDPatch Utility-out-0.05-np-1-nc-12-mv-0.5 Utility-out-0.05-np-1-nc-12-mv-0.5 "${root_patch}/Utility/${python_fname}-0-RUN.py" True