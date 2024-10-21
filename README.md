# CatDB: Data-catalog-guided, LLM-based Generation of Data-centric ML Pipelines

![Overview](images/workflow.png)

**Overview:** CatDB a comprehensive, LLM-guided generator of data-centric ML pipelines that utilizes available data catalog information. We incorporate data profiling information and user descriptions into a chain of LLM prompts for data cleaning/augmentation, feature engineering, and model selection. Additionally, we devise a robust framework for managing the LLM interactions and handling errors through pipeline modifications and a knowledge base of error scenarios.


Resource        | Links
----------------|------
**Quick Start** | [Install and Quick Start]()
**Benchmarks:** | [Execute experiments with a range of datasets and compare the results to baselines]()


## Installation
* Clone CatDB Repository:
    ```
    git clone https://github.com/CoDS-GCS/CatDB.git
    ```
* Install Requirenments:
    ```
    cd CatDB/src/python/main
    python -m venv venv
    source venv/bin/activate 
    python -m pip install -r requirements.txt
    ```
## Configuration
* Setup your LLM service API Keys ...

## Run CatDB
* Prepare Dataset
* Profile Data
* Run CatDB to Generating ML Pipeline:
    ```
    python main.py \
        --llm-model \ 
        --metadata-path \
        --root-data-path \
        --catalog-path \
        --prompt-representation-type \
        --prompt-number-iteration \
        --prompt-number-iteration-error \
        --output-path \
        --result-output-path \
        --error-output-path
    ```

