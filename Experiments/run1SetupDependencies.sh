#!/bin/bash

# sudo add-apt-repository ppa:deadsnakes/ppa
# sudo apt-get update
# sudo apt-get install -y openjdk-11-jdk-headless

# sudo apt-get install -y clang++
# sudo apt-get install -y python3-dev
# sudo apt-get install python3.9-venv # required for automlbenchmark
# sudo apt-get install python3.10-venv # required for CAAFE 
# sudo apt install -y python3-pip
# sudo pip install -y virtualenv
# sudo apt-get install -y wget
# sudo apt-get install -y unzip

# root_path="$(pwd)"

# # Setup Apache Spark 
# mkdir -p "${root_path}/dependencies"
# cd "${root_path}/dependencies"

# rm -rf spark-3.5.0-bin-hadoop3.tgz # clean-up
# rm -rf spark-3.5.0-bin-hadoop3 # clean-up
# wget https://dlcdn.apache.org/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
# tar -xvzf spark-3.5.0-bin-hadoop3.tgz

# # setup Anaconda
# rm -rf Anaconda3-2023.09-0-Linux-x86_64 # clean-up
# wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
# sha256sum Anaconda3-2023.09-0-Linux-x86_64.sh
# bash Anaconda3-2023.09-0-Linux-x86_64.sh

# setup gcloud for FineTune Google
#rm -rf tmprepo
mkdir -p tmprepo #clean-up

cd tmprepo
#curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
#tar -xf google-cloud-cli-linux-x86_64.tar.gz
#./google-cloud-sdk/install.sh
./google-cloud-sdk/bin/gcloud init

cd ${root_path}

