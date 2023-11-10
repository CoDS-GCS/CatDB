#!/bin/bash

#cleanup
root_path="$(pwd)"
path="$(pwd)/setup"
rm -rf "$path/Baselines"
mkdir -p "$path/Baselines"

echo $root_path

cd $root_path
cd ..

# build and setup Python baseline
cp -r Baselines/src/* ${path}"/Baselines/"

# cd ${path}"/Baselines/"
# pip install -r requirements.txt # install requirements
