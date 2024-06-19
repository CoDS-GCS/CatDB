from automl.AutoML import AutoML as CatDBAutoML
from util.Config import Config
from util.Data import Dataset
from sklearn.metrics import r2_score, mean_squared_error

import os
import time
import pandas as pd
import tempfile
from sklearn.metrics import roc_auc_score
from autogluon.tabular import TabularPredictor, TabularDataset
import autogluon.core.metrics as metrics
from autogluon.tabular.version import __version__


class AutogluonAutoML(CatDBAutoML):
    def __init__(self, dataset: Dataset, config: Config, *args, **kwargs):
        CatDBAutoML.__init__(self, dataset=dataset, config=config)

    def run(self):
        print(f"\n**** AutoGluon [v{__version__}] ****\n")

        sort_metric = None
        if self.dataset.task_type == 'binary':
            sort_metric = metrics.roc_auc
        elif self.dataset.task_type == 'multiclass':
            sort_metric = metrics.roc_auc_ovo_macro
        elif self.dataset.task_type == 'regression':
            sort_metric = metrics.r2

        models_dir = tempfile.mkdtemp() + os.sep

        time_start = time.time()
        predictor = TabularPredictor(
            label=self.dataset.target_attribute,
            eval_metric=sort_metric,
            path=models_dir,
            problem_type=self.dataset.task_type,
            verbosity=False
        ).fit(
            train_data=self.dataset.train_path,
            time_limit=self.config.max_runtime_seconds)

        test_data = TabularDataset(self.dataset.test_path)
        train_data = TabularDataset(self.dataset.train_path)

        time_end = time.time()
        time_execute = time_end - time_start

        if self.dataset.task_type == "binary":
            result_test = predictor.evaluate(test_data, silent=True)
            result_train = predictor.evaluate(train_data, silent=True)

            self.log_results.train_auc = result_train["roc_auc"]
            self.log_results.train_accuracy = result_train["accuracy"]
            self.log_results.train_f1_score = result_train["f1"]

            self.log_results.test_auc = result_test["roc_auc"]
            self.log_results.test_accuracy = result_test["accuracy"]
            self.log_results.test_f1_score = result_test["f1"]

        elif self.dataset.task_type == "multiclass":
            result_test = predictor.evaluate(test_data, silent=True)
            result_train = predictor.evaluate(train_data, silent=True)

            y_train = train_data[self.dataset.target_attribute]
            y_test = pd.DataFrame(test_data[self.dataset.target_attribute])

            predictions_test = predictor.predict_proba(test_data, as_multiclass=True)
            predictions_train = predictor.predict_proba(train_data, as_multiclass=True)

            self.log_results.train_auc_ovr = roc_auc_score(y_train, predictions_train, multi_class='ovr')
            self.log_results.train_auc_ovo = result_train["roc_auc_ovo_macro"]
            self.log_results.train_accuracy = result_train["accuracy"]

            self.log_results.test_auc_ovr = roc_auc_score(y_test, predictions_test, multi_class='ovr')
            self.log_results.test_auc_ovo = result_test["roc_auc_ovo_macro"]
            self.log_results.test_accuracy = result_test["accuracy"]

        elif self.dataset.task_type == "regression":
            y_train_pred = predictor.predict(train_data, as_pandas=True)
            y_test_pred = predictor.predict(test_data, as_pandas=True)

            y_train = train_data[self.dataset.target_attribute]
            y_test = pd.DataFrame(test_data[self.dataset.target_attribute])

            self.log_results.train_r_squared = r2_score(y_train, y_train_pred)
            self.log_results.train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)

            self.log_results.test_r_squared = r2_score(y_test, y_test_pred)
            self.log_results.test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

        self.log_results.status = "True"
        self.log_results.time_execution = time_execute
        self.log_results.config = "Autogluon"
        self.log_results.save_results(result_output_path=self.config.output_path)

