#!/bin/bash
root_path="$(pwd)"
data_path="${root_path}/data"

# Enable venv for download datasets and resources_m
cd ${data_path}
rm -rf venv
python -m venv venv
source venv/bin/activate

# install packages
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

ls -hl
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python DownloadOpenMLDatasetsByTaskID.py ${data_path}

mv "${data_path}/catdb_openml.yaml" "${root_path}/setup/automlbenchmark/resources/benchmarks/"

cd ${root_path}