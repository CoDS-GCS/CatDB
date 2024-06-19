from automl.AutoML import AutoML as CatDBAutoML, result
from util.Config import Config
from util.Data import Dataset, reader_CSV
import pandas as pd

import time
import autosklearn
from autosklearn.estimators import AutoSklearnRegressor
from autosklearn.experimental.askl2 import AutoSklearn2Classifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.metrics import r2_score, mean_squared_error
from packaging import version
import sklearn.metrics
import math
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_numeric_dtype
import warnings


class AutoSklearnAutoML(CatDBAutoML):
    def __init__(self, dataset: Dataset, config: Config, *args, **kwargs):
        CatDBAutoML.__init__(self, dataset=dataset, config=config)

    def run(self):
        askl_version = version.parse(autosklearn.__version__)
        print(f"\n**** Auto-Sklearn [v{askl_version}] ****\n")

        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=DeprecationWarning)

        time_start = time.time()
        train_data = reader_CSV(self.dataset.train_path)
        test_data = reader_CSV(self.dataset.test_path)

        X_train = train_data.drop(columns=[self.dataset.target_attribute])
        y_train = train_data[self.dataset.target_attribute]

        X_test = test_data.drop(columns=[self.dataset.target_attribute])
        y_test = test_data[self.dataset.target_attribute]

        if is_numeric_dtype(train_data[self.dataset.target_attribute]) == False or is_numeric_dtype(test_data[self.dataset.target_attribute]) == False:
            y_all = pd.concat([y_train, y_test])
            le = LabelEncoder()
            trained_le = le.fit(y_all)
            y_train = trained_le.transform(y_train)
            y_test = trained_le.transform(y_test)

        constr_params = {}
        ml_memory_limit = max(
            min(
                self.config.jvm_memory / self.config.nthreads,
                math.ceil(self.config.jvm_memory / self.config.nthreads)
            ),
            3072  # 3072 is autosklearn default and we use it as a lower bound
        )
        constr_params["memory_limit"] = ml_memory_limit

        if self.dataset.task_type == "binary" or self.dataset.task_type == "multiclass":
            estimator = AutoSklearn2Classifier
        else:
            estimator = AutoSklearnRegressor

        constr_params["time_left_for_this_task"] = self.config.max_runtime_seconds
        constr_params["n_jobs"] = self.config.nthreads
        constr_params["seed"] = self.config.seed
        automl = estimator(**constr_params)

        automl.fit(X=X_train, y=y_train, X_test=X_test, y_test=y_test)

        y_train_pred = automl.predict(X_train)
        y_test_pred = automl.predict(X_test)

        time_end = time.time()
        time_execute = time_end - time_start

        if self.dataset.task_type == "binary":
            self.log_results.train_accuracy = sklearn.metrics.accuracy_score(y_train, y_train_pred)
            self.log_results.train_f1_score = f1_score(y_train, y_train_pred, average='weighted')
            self.log_results.train_auc = roc_auc_score(y_train, automl.predict_proba(X_train)[:, 1])

            self.log_results.test_accuracy = sklearn.metrics.accuracy_score(y_test, y_test_pred)
            self.log_results.test_f1_score = f1_score(y_test, y_test_pred, average='weighted')
            self.log_results.test_auc = roc_auc_score(y_test, automl.predict_proba(X_test)[:, 1])

        elif self.dataset.task_type == "multiclass":
            self.log_results.train_accuracy = sklearn.metrics.accuracy_score(y_train, y_train_pred)
            self.log_results.train_log_loss = log_loss(y_train, automl.predict_proba(X_train))
            self.log_results.train_auc_ovo = roc_auc_score(y_train, automl.predict_proba(X_train), multi_class='ovo')
            self.log_results.train_auc_ovr = roc_auc_score(y_train, automl.predict_proba(X_train), multi_class='ovr')

            self.log_results.test_accuracy = sklearn.metrics.accuracy_score(y_test, y_test_pred)
            self.log_results.test_log_loss = log_loss(y_test, automl.predict_proba(X_test))
            self.log_results.test_auc_ovo = roc_auc_score(y_test, automl.predict_proba(X_test), multi_class='ovo')
            self.log_results.test_auc_ovr = roc_auc_score(y_test, automl.predict_proba(X_test), multi_class='ovr')

        else:
            self.log_results.train_rmse  = mean_squared_error(y_train, y_train_pred, squared=False)
            self.log_results.train_r_squared = r2_score(y_train, y_train_pred)

            self.log_results.test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
            self.log_results.test_r_squared = r2_score(y_test, y_test_pred)

        self.log_results.number_iteration = self.config.iteration
        self.log_results.status = "True"
        self.log_results.time_execution = time_execute
        self.log_results.config = "AutoSklearn"
        self.log_results.save_results(result_output_path=self.config.output_path)
