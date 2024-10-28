## Benchmark

**Source Code Info:**
 * Repository: CatDB (<https://github.com/CoDS-GCS/CatDB>)
 * Programming Language: Python 3.10 & 3.9, Java  
 * Packages/Libraries Needed: JDK 11, Python, Git, Maven, pdflatex, unzip, unrar, xz-utils

**Datasets Used:**
\#|Dataset | URL | Download Link
--|--------|-----|---
1|Diabetes       | OpenML: DatasetID \#37                         | [download](https://www.openml.org/search?type=data&sort=runs&status=active&id=37)
2|Breast-w       | OpenML: DatasetID \#15			  | [download](https://www.openml.org/search?type=data&sort=runs&status=active&id=15)	
3|Tic-Tac-Toe    | OpenML: DatasetID \#50                         | [download](https://www.openml.org/search?type=data&sort=runs&status=active&id=50)
4|Credit-g       | OpenML: DatasetID \#31                         | [download](https://www.openml.org/search?type=data&sort=runs&status=active&id=31)
5|Nomao          | OpenML: DatasetID \#1486                       | [download](https://www.openml.org/search?type=data&sort=runs&status=active&id=1486)
6|Walking        | OpenML: DatasetID \#1509                       | [download](https://www.openml.org/search?type=data&sort=runs&status=active&id=1509)
7|CMC            | OpenML: DatasetID \#23                         | [download](https://www.openml.org/search?type=data&sort=runs&status=active&id=23)
8|Gas-Drift      | OpenML: DatasetID \#1476                       | [download](https://www.openml.org/search?type=data&sort=runs&status=active&id=1476)
9|Volkert        | OpenML: DatasetID \#41166                      | [download](https://www.openml.org/search?type=data&sort=runs&status=active&id=41166)  
10|Bike-Sharing  | OpenML: DatasetID \#44048                      | [download](https://www.openml.org/search?type=data&sort=runs&status=active&id=44048) 
11|NYC           | OpenML: DatasetID \#44065                      | [download](https://www.openml.org/search?type=data&sort=runs&status=active&id=44065)
12|House-Sales   | OpenML: DatasetID \#44051                      | [download](https://www.openml.org/search?type=data&sort=runs&status=active&id=44051)
13|IMDB          | https://relational-data.org/dataset/IMDb       | [download](https://relational-data.org/dataset/IMDb)
14|Accidents     | https://relational-data.org/dataset/Accidents  | [download](https://relational-data.org/dataset/Accidents)
15|Financial     | https://relational-data.org/dataset/Financial  | [download](https://relational-data.org/dataset/Financial)
16|Airline       | https://relational-data.org/dataset/Airline    | [download](https://relational-data.org/dataset/Airline)
17|Yelp          | https://relational-data.org/dataset/Yelp       | [download](https://relational-data.org/dataset/Yelp)
18|Wifi          | Local                                          | [download](https://github.com/CoDS-GCS/CatDB/blob/main/Experiments/data/Accidents.zip) 
19|EU IT         | Local                                          | [download](https://github.com/CoDS-GCS/CatDB/blob/main/Experiments/data/EU-IT.zip)
20|Survey        | Local                                          | [download](https://github.com/CoDS-GCS/CatDB/blob/main/Experiments/data/Midwest-Survey.zip)
21|Etailing      | Local                                          | [download](https://github.com/CoDS-GCS/CatDB/blob/main/Experiments/data/Etailing.zip)
22|Utility       | Local                                          | [https://github.com/CoDS-GCS/CatDB/blob/main/Experiments/data/Utility.zip)

**Note:**

All datasets will be downloaded automatically.



**Hardware and Software Info:** We ran all experiments on a server node (VM) with an Intel Core CPU (with 32 vcores) and 150 GB of DDR4 RAM. The software stack consisted of Ubuntu 22.04, OpenJDK 11 (for Java baselines), and Python 3.10 (for Python baselines).

**Setup and Experiments:** The repository is pre-populated with the paper's experimental results (`./results`), individual plots (`./plots`), and SystemDS source code. The entire experimental evaluation can be run via `./runAll.sh`, which deletes the results and plots and performs setup, dataset download, dataset preparation, dataset generating, local experiments, and plotting. However, for a more controlled evaluation, we recommend running the individual steps separately.
```
./run1SetupDependencies.sh;
./run2SetupBaseLines.sh;
./run3DownloadData.sh;
./run4PrepareData.sh;
./run5LocalExperiments.sh;
./run6PlotResults.sh; 
```

The `./run1SetupDependencies.sh` script installs all the required dependencies. Here is a brief overview of each dependency and its purpose:

* **JDK 11**: for Java-based baselines (H2O AutoML)
* **unzip**, **unrar**, and **xz-utils**: for decompressing datasets
* **python3.9 & 3.10**: for python-based baselines
* **pdflatex >2021**: for result visualization

The `./run2SetupBaseLines.sh` script will automatically compile Java, and Python based implementations and set up the runnable apps in the `Setup` directory. There is no need for manual effort in this process.

We manage our datasets using two scripts: `./run3DownloadData.sh` and `./run4GenerateData.sh`.

* In the `./run3DownloadData.sh` script, we automatically download all datasets used in the experiments. The refined format of these datasets is then moved into the `data` directory.

* The `./run4GenerateData.sh` script is responsible for generating missing values and outlier existing ones.

* Missing Value and Outlier generated datasets: Utility and Volkert

The `./run5LocalExperiments.sh` script is responsible for running all experiments. For each experiment, we have planned to execute it five times and store the experimental results in the `results` directory.

Since we run experiments five times in the `./run6PlotResults.sh` script, we follow the following process:

* Plot the averaged results using LaTeX's tikzpicture and store the plots in the `plots` directory.


**Last Update:** Oct 27, 2024 (draft version)