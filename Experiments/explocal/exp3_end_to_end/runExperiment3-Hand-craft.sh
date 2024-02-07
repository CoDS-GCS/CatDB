#!/bin/bash


root_path="$(pwd)"
baselines_path="${root_path}/baselines"
cd ${baselines_path}

source venv/bin/activate

python diabetes_kaggle.py