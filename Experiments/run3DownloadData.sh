#!/bin/bash

data_path="$(pwd)/data"
cd ${data_path}

rm -rf venv
python -m venv venv
source venv/bin/activate

ls -hl
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python DownloadOpenMLDatasetsByTaskID.py ${data_path}
