#!/bin/bash

#cleanup
root_path="$(pwd)"
path="$(pwd)/setup"
mkdir -p "$path/Baselines"

cd $path

# # Setup kglidsplus
# #################
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
catdb_path="${path}/Baselines/CatDB/"
rm -rf ${catdb_path}
mkdir -p ${catdb_path}

cd ${root_path}
cd ..
cp -r src/python/main/* ${catdb_path}
cd ${catdb_path}

rm -rf venv 
python3.10 -m venv venv
source venv/bin/activate

# Then install the dependencies:
python3.10 -m pip install --upgrade pip
pip install -r requirements.txt

# Prepare Config
################
config_path="${path}/config/"
cd ${config_path}
rm -rf venv

python3.10 -m venv venv
source venv/bin/activate

#Then install the dependencies:
python3.10 -m pip install --upgrade pip
pip install -r requirements.txt

# Setup CAAFE
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


## Setup AutoML
automl_path="${path}/Baselines/AutoML/"
rm -rf ${automl_path}
mkdir -p ${automl_path}

cd ${root_path}
cp -r baselines/* "${automl_path}/"


# ## Install Auto Sklearn
cd ${automl_path}
cd AutoSklearnAutoML
rm -rf venv 
python3.9 -m venv venv
source venv/bin/activate
python3.9 -m pip install --upgrade pip
python3.9 -m pip install --no-cache-dir -r requirements.txt
pip uninstall -y numpy
pip install --no-cache-dir numpy==1.26.4
pip unistall -y pandas
pip install --no-cache-dir -U pandas==1.5.3

# ## Install H2O AutoML
cd ${automl_path}
cd H2OAutoML
rm -rf venv 
python3.10 -m venv venv
source venv/bin/activate
python3.10 -m pip install --upgrade pip
python3.10 -m pip install -r requirements.txt

# ## Install Flaml AutoML
cd ${automl_path}
cd FlamlAutoML
rm -rf venv 
python3.10 -m venv venv
source venv/bin/activate
python3.10 -m pip install --no-cache-dir --upgrade pip
python3.10 -m pip install --no-cache-dir -r requirements.txt

# ## Install Autogluon  AutoML
cd ${automl_path}
cd AutogluonAutoML
rm -rf venv 
python3.10 -m venv venv
source venv/bin/activate
python3.10 -m pip install --upgrade pip
python3.10 -m pip install -r requirements.txt

# ## Install AutoGen
cd ${automl_path}
cd AutoGenAutoML
rm -rf venv 
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# # Setup AIDE
baselines_path="${path}/Baselines"
cd ${baselines_path}
rm -rf aideml
git clone --branch CatDB https://github.com/fathollahzadeh/aideml.git

cd aideml
rm -rf venv
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .


### Setup SAGA
baselines_path="${path}/Baselines"
cd ${baselines_path}
rm -rf SAGA

cd ${root_path}
cp -r baselines/SAGA "${baselines_path}/"

cd "${baselines_path}/SAGA"

rm -rf venv
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

 rm -rf systemds
 git clone https://github.com/apache/systemds.git

cd systemds
mvn clean package -P distribution
cd ..

rm -rf SystemDS.jar
mv systemds/target/SystemDS.jar "${baselines_path}/SAGA"
rm -rf lib
mv systemds/target/lib/ "${baselines_path}/SAGA/"


# ## Setup Learn2Clean
baselines_path="${path}/Baselines"
cd ${baselines_path}
rm -rf Learn2Clean

cd ${root_path}
cp -r baselines/Learn2Clean "${baselines_path}/"

cd "${baselines_path}/Learn2Clean"
rm -rf venv
python3.8 -m venv venv
source venv/bin/activate

git clone -b CatDB https://github.com/fathollahzadeh/Learn2Clean.git
cd Learn2Clean/python-package
python setup.py install
pip install -r requirements.txt

# ## Setup Augmentation
baselines_path="${path}/Baselines"
cd ${baselines_path}
rm -rf Augmentation

cd ${root_path}
cp -r baselines/Augmentation "${baselines_path}/"

cd "${baselines_path}/Augmentation"

rm -rf venv
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt