#!/bin/bash

#cleanup
root_path="$(pwd)"
path="$(pwd)/setup"
mkdir -p "$path/Baselines"

cd $path

# unzip venv template
# cd "${root_path}/setup/config"
# venv_path="${root_path}/setup/config/venv"
# unzip "${venv_path}.zip"

# cd $path

# Setup kglidsplus
#################
# cd "${path}/Baselines"
# rm -rf kglidsplus
# git clone git@github.com:CoDS-GCS/kglidsplus.git

# cd kglidsplus
# conda create -n kglidsplus python=3.8 -y
# eval "$(conda shell.bash hook)"
# conda activate kglidsplus
# python3.8 -m pip install --upgrade pip
# pip install -r requirements.txt

# cp "${root_path}/setup/config/kglidsplus_main.py"  "${path}/Baselines/kglidsplus/kg_governor/data_profiling/src/"


# Setup CatDB
#############
# catdb_path="${path}/Baselines/CatDB/"
# # rm -rf ${catdb_path}
# mkdir -p ${catdb_path}

# cd ${root_path}
# cd ..
# cp -r src/python/main/* ${catdb_path}
# cd ${catdb_path}

# rm -rf venv 
# #cp -r ${venv_path} ${catdb_path} 
# python -m venv venv
# source venv/bin/activate

# # Then install the dependencies:
# python -m pip install --upgrade pip
# pip install torchvision 
# python -m pip install -r requirements.txt
# pip install pipreqs

# Prepare Config
################
# config_path="${path}/config/"
# cd ${config_path}
# rm -rf venv

# python -m venv venv
# #cp -r ${venv_path} ${catdb_path} 
# source venv/bin/activate

# #Then install the dependencies:
# python -m pip install --upgrade pip
# python -m pip install -r requirements.txt

# Setup Hand-craft baseline
################
# baselines_path="${root_path}/baselines/"
# cd ${baselines_path}
# rm -rf venv
# python -m venv venv
# source venv/bin/activate

# # Then install the dependencies:
# pip install pipreqs
# pipreqs --force --mode no-pin ${baselines_path}
# python -m pip install --upgrade pip
# python -m pip install -r requirements.txt



# Setup CAFFE
#############
baselines_path="$path/Baselines"
cd ${baselines_path}
rm -rf CAAFE
git clone --branch catdb https://github.com/fathollahzadeh/CAAFE.git

cd CAAFE
rm -rf venv
python3.9 -m venv venv
source venv/bin/activate
python3.9 -m pip install --upgrade pip
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
python3.9 -m pip install -r requirements.txt
