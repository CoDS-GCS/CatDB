#!/bin/bash

root_path="$(pwd)"
data_path="${root_path}/data"
config_path="${root_path}/setup/config/"

mkdir -p ${data_path}
cd ${config_path}
source venv/bin/activate

#python DownloadOpenMLDatasetsByDatasetID.py --data-out-path ${data_path}

python GenerateDataMissingValues.py --data-in-path "${config_path}/datasets" --data-out-path ${data_path}
cd ${root_path}


# Multitable datasets:
# classification
# 1. Accident, Multiclass: 
# Count of tables: 3
# Count of rows: 1,453,650
# Count of columns: 43
# URL: https://relational-data.org/dataset/Accidents

# 2: IMDB, binary classification:
# Count of tables: 7
# Count of rows: 5,694,919
# Count of columns: 21
# URL: https://relational-data.org/dataset/IMDb

# 3: Airline, multiclass classification:
# Count of tables: 19
# Count of rows: 448,156
# Count of columns: 119
# URL: https://relational-data.org/dataset/Airline

# 4: Hockey
# Count of tables: 22
# Count of rows: 96,403
# Count of columns: 300
# URL: https://relational-data.org/dataset/Hockey

# 5: Financial
# Count of tables: 8
# Count of rows: 1,090,086
# Count of columns: 55
# URL: https://relational-data.org/dataset/Financial

# regression:
# 1. Walmart
# Count of tables: 4
# Count of rows: 4,628,497
# Count of columns: 27
# URL: https://relational-data.org/dataset/Walmart

# 2. Lahman
# Count of tables: 25
# Count of rows: 470,225
# Count of columns: 353
# URL:https://relational-data.org/dataset/Lahman

# 3. Yelp

