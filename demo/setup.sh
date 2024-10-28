#!/bin/bash

# Create a setup path and install CatDB
mkdir -p catdb-setup
cp -r /content/CatDB/src/python/main/* /content/catdb-setup
cd /content/catdb-setup/
apt install python3.10-venv
rm -rf venv # clean-up last env 
python3.10 -m venv venv
source venv/bin/activate

# Then install the dependencies:
python3.10 -m pip install --upgrade pip
pip install torchvision 
python3.10 -m pip install -r requirements.txt

# Prepare demo data
cd /content/demo/
apt install -y unzip
unzip catalog.zip
unzip data.zip
mkdir catdb-results

cp /content/demo/demo.py /content/catdb-setup/

# Clean-up
rm -rf /content/CatDB