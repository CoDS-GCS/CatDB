#!/bin/bash

sudo apt update
sudo apt-get install -y openjdk-11-jdk-headless

sudo apt-get install clang++

# Python Environment 
# rm -rf envCatDB # clean up
# apt install -y python3-pip virtualenv
# virtualenv -p python3 envCatDB #Create an environment: envCatDB
# source envCatDB/bin/activate #Active environment: envCatDB