#!/bin/bash

sudo apt update
sudo apt-get install -y openjdk-11-jdk-headless

sudo apt-get install clang++
sudo apt-get install python3-dev
sudo apt-get install wget

root_path="$(pwd)"

# Setup Apache Spark 
mkdir -p "${root_path}/dependencies"
cd "${root_path}/dependencies"

wget https://www.apache.org/dyn/closer.lua/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
tar -xvzf spark-3.5.0-bin-hadoop3.tgz

# setup Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
sha256sum Anaconda3-2023.09-0-Linux-x86_64.sh
bash Anaconda3-2023.09-0-Linux-x86_64.sh

cd ${root_path}