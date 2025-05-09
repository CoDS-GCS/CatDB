# CatDB: Data-catalog-guided, LLM-based Generation of Data-centric ML Pipelines

![Overview](images/workflow.png)

**Overview:** CatDB a comprehensive, LLM-guided generator of data-centric ML pipelines that utilizes available data catalog information. We incorporate data profiling information and user descriptions into a chain of LLM prompts for data cleaning/augmentation, feature engineering, and model selection. Additionally, we devise a robust framework for managing the LLM interactions and handling errors through pipeline modifications and a knowledge base of error scenarios.


Resource        | Links
----------------|------
**Quick Start** | [Install and Quick Start](#Installation)
**PVLDB v18 2025 Reproducibility:** | [Execute experiments with a range of datasets and compare the results to baselines](https://github.com/CoDS-GCS/CatDB/tree/main/Experiments)


## Installation
* Install Data Profiling:
    ```
    pip install -U git+https://github.com/CoDS-GCS/CatDB.git@dataprofiling --quiet 
    ```
* Install CatDB:
    ```
    pip install -U git+https://github.com/CoDS-GCS/CatDB.git --quiet 
    ```
* Install and Config Apache Spark:
    ```
    wget -q https://archive.apache.org/dist/spark/spark-3.5.3/spark-3.5.3-bin-hadoop3.tgz -O spark-3.5.3-bin-hadoop3.tgz
    tar xf spark-3.5.3-bin-hadoop3.tgz
    os.environ["SPARK_HOME"] = "spark-3.5.3-bin-hadoop3/"
    os.environ["PYSPARK_PYTHON"] = "python"
    ```    
## Quickstart
* Generate ML pipeline for `adult` dataset (`binary` classification with `income` target attribute):
```python
from catdb import config, create_report, generate_pipeline, prepare_dataset
from dataprofiling import build_catalog

cfg = config(model="gpt-4o", API_key="...", iteration=5)

data = prepare_dataset(path="adult.csv", task_type="binary", target_attribute="income")

catalog = build_catalog(data=data, categorical_ratio=0.05, n_workers=1 ,max_memory=10)

pipeline = generate_pipeline(catalog, cfg)

```

## Contributions
We encourage contributions and bug fixes, please don't hesitate to open a PR or create an issue if you face any bugs.

## Contact
For any questions please contact us:

* **Saeed Fathollahzadeh**: saeed.fathollahzadeh@concordia.ca, s.fathollahzadeh@gmail.com
* **Essam Mansour** : essam.mansour@concordia.ca
* **Matthias Boehm**: matthias.boehm@tu-berlin.de
