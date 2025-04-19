## Reproducibility Submission PVLDB v18  Paper #ID: 995

**Source Code Info:**
 * Repository: CatDB (<https://github.com/CoDS-GCS/CatDB>)
 * Programming Language: Python 3.10 & 3.9, Java  
 * Packages/Libraries Needed: JDK 11, Python, Git, Maven, pdflatex, unzip, unrar, xz-utils


**Hardware and Software Info:** We ran all experiments on a server node (VM) with an Intel Core CPU (with 32 vcores) and 150 GB of DDR4 RAM. The software stack consisted of Ubuntu 22.04, OpenJDK 11 (for Java baselines), and Python 3.10 (for Python baselines).

**Setup and Experiments:** The repository is pre-populated with the paper's experimental results (`./results`), individual plots (`./plots`), and SystemDS source code. The entire experimental evaluation can be run via `./runAll.sh`, which deletes the results and plots and performs setup, dataset download, dataset preparation, dataset generating, local experiments, and plotting. However, for a more controlled evaluation, we recommend running the individual steps separately.

### Step 1: Dependency Setup
--- 
The `./run1SetupDependencies.sh` script installs all the required dependencies. Here is a brief overview of each dependency and its purpose:

* **JDK 11**: for Java-based baselines (H2O AutoML)
* **unzip**, **unrar**, and **xz-utils**: for decompressing datasets
* **python3.9 & 3.10**: for python-based baselines
* **pdflatex =2021**: for result visualization


### Step 2: Baselines Setup and Config
--- 
The `./run2SetupBaseLines.sh` script will automatically compile Java, and Python based implementations and set up the runnable apps in the `Setup` directory. There is no need for manual effort in this process.

For LLM-based baselines, you need to set API keys for the LLM services (OpenAI, Google Gemini, and Llama). For each service, create an API key using the following links:

* **OpenAI**: [https://platform.openai.com/](https://platform.openai.com/)
* **Google Gemini**:[https://aistudio.google.com/](https://aistudio.google.com/)
* **Groq (Llama)**: [https://console.groq.com/](https://console.groq.com/)

The API keys must be set in the following path:

* **CatDB Setup Path**:
```
# Path:
Experiments/stup/Baselines/CatDB/APIKeys.yaml

# Content:
---

- llm_platform: OpenAI
  key_1: 'YOUR KEY'

- llm_platform: Meta
  key_1: 'YOUR KEY'

- llm_platform: Google
  key_1 : 'YOUR KEY'
```

* **CAAFE Setup Path**: setup in OS path
```
export OPENAI_API_KEY_1=<YOUR KEY>
export GROQ_API_KEY_1=<YOUR KEY>
export GOOGLE_API_KEY_1=<YOUR KEY>
```

* **AIDE Setup Path**:
```
# Path:
Experiments/setup/Baselines/aideml/APIKeys.yaml

# Content:
---

- llm_platform: OpenAI
  key_1: 'YOUR KEY'

- llm_platform: Meta
  key_1: 'YOUR KEY'

- llm_platform: Google
  key_1 : 'YOUR KEY'
```

* **AutoGen Setup Path**:
```
# Path:
Experiments/setup/Baselines/AutoML/AutoGenAutoML/OAI_CONFIG_LIST

# Content:
[
    {
        "model": "gpt-4o",
        "api_key":"YOUR KEY"
    },
    {
        "model": "llama-3.1-70b-versatile",
        "api_key": "YOUR KEY",
        "api_type": "groq"
    },
    {
        "model": "gemini-1.5-pro",
        "api_key": "YOUR KEY",
        "api_type": "google"
    }
]
```

### Step 3: Download and Prepare Datasets
--- 
We manage our datasets using two scripts: `./run3DownloadData.sh` and `./run4PrepareData.sh`.

* In the `./run3DownloadData.sh` script, we automatically download all datasets used in the experiments. The refined format of these datasets is then moved into the `data` directory.

* The `./run4PrepareData.sh` script decompresses both the local datasets and the data catalog files. It also splits the data according to the paper's settings, using a `70/30` split for training and test datasets.

**Datasets Used:**
\#|Dataset | URL | Download Link
--|--------|-----|---
1|Wifi           | Local                                          | [download](https://github.com/CoDS-GCS/CatDB/blob/main/Experiments/data/Accidents.zip) 
2|Diabetes       | OpenML: DatasetID \#37                         | [download](https://www.openml.org/search?type=data&sort=runs&status=active&id=37)
3|Tic-Tac-Toe    | OpenML: DatasetID \#50                         | [download](https://www.openml.org/search?type=data&sort=runs&status=active&id=50)
4|IMDB           | https://relational-data.org/dataset/IMDb       | [download](https://relational-data.org/dataset/IMDb)
5|KDD98          |OpenML: DatasetID \#42343                       | [download](https://www.openml.org/search?type=data&sort=runs&status=active&id=42343)
6|Walking        | OpenML: DatasetID \#1509                       | [download](https://www.openml.org/search?type=data&sort=runs&status=active&id=1509)
7|CMC            | OpenML: DatasetID \#23                         | [download](https://www.openml.org/search?type=data&sort=runs&status=active&id=23)
8|EU IT          | Local                                          | [download](https://github.com/CoDS-GCS/CatDB/blob/main/Experiments/data/EU-IT.zip)
9|Survey         | Local                                          | [download](https://github.com/CoDS-GCS/CatDB/blob/main/Experiments/data/Midwest-Survey.zip)
10|Etailing      | Local                                          | [download](https://github.com/CoDS-GCS/CatDB/blob/main/Experiments/data/Etailing.zip)
11|Accidents     | https://relational-data.org/dataset/Accidents  | [download](https://relational-data.org/dataset/Accidents)
12|Financial     | https://relational-data.org/dataset/Financial  | [download](https://relational-data.org/dataset/Financial)
13|Airline       | https://relational-data.org/dataset/Airline    | [download](https://relational-data.org/dataset/Airline)
14|Gas-Drift      | OpenML: DatasetID \#1476                      | [download](https://www.openml.org/search?type=data&sort=runs&status=active&id=1476)
15|Volkert        | OpenML: DatasetID \#41166                     | [download](https://www.openml.org/search?type=data&sort=runs&status=active&id=41166)
16|Yelp          | https://relational-data.org/dataset/Yelp       | [download](https://relational-data.org/dataset/Yelp)
17|Bike-Sharing  | OpenML: DatasetID \#44048                      | [download](https://www.openml.org/search?type=data&sort=runs&status=active&id=44048) 
18|Utility       | Local                                          | [download](https://github.com/CoDS-GCS/CatDB/blob/main/Experiments/data/Utility.zip)
19|NYC           | OpenML: DatasetID \#44065                      | [download](https://www.openml.org/search?type=data&sort=runs&status=active&id=44065)
20|House-Sales   | OpenML: DatasetID \#44051                      | [download](https://www.openml.org/search?type=data&sort=runs&status=active&id=44051)


### Step 4: Run Experiments
--- 
The `./run5LocalExperiments.sh` script is responsible for running all experiments. For each experiment, we have planned to execute it five times and store the experimental results in the `results` directory.


### Step 5: Plotting Diagrams
--- 
Since we run experiments five times in the `./run6PlotResults.sh` script, we follow the following process:

* Plot the averaged results using LaTeX's tikzpicture and store the plots in the `plots` directory.

### All Scripts: ```./runAll.sh```
--- 

```
./run1SetupDependencies.sh;
./run2SetupBaseLines.sh;
./run3DownloadData.sh;
./run4PrepareData.sh;
./run5LocalExperiments.sh;
./run6PlotResults.sh; 
```

--- 
**Last Update:** April 19, 2025 (draft version)