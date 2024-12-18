from automl.AutoML import AutoML as CatDBAutoML
from util.Config import Config
from util.Data import Dataset
import os
import time
import pandas as pd
import tempfile

class AideML(CatDBAutoML):
    def __init__(self, dataset: Dataset, config: Config, *args, **kwargs):
        CatDBAutoML.__init__(self, dataset=dataset, config=config)

        
