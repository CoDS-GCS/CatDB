#!/bin/bash
 
exp_path="$(pwd)" 
cd "${exp_path}/setup/config/"
source venv/bin/activate
python MergeCatDBAndGenerateAutoMLConfig.py "${exp_path}/results" "${exp_path}/catdb-results" "${exp_path}/explocal/exp2_micro_benchmark"
