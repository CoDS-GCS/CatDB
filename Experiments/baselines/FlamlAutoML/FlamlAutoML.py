from automl.AutoML import AutoML as CatDBAutoML
from util.Config import Config
from util.Data import Dataset, reader_CSV
from sklearn.metrics import f1_score, roc_auc_score, log_loss
from sklearn.metrics import r2_score, mean_squared_error
import time
import re

import os
from flaml import AutoML, __version__


class FlamlAutoML(CatDBAutoML):
    def __init__(self, dataset: Dataset, config: Config, *args, **kwargs):
        CatDBAutoML.__init__(self, dataset=dataset, config=config)

    
    def run(self):
        print(f"\n**** FLAML [v{__version__}] ****\n")

        time_start = time.time()
        train_data = reader_CSV(self.dataset.train_path)
        test_data = reader_CSV(self.dataset.test_path)

#         import re
# df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

        X_train = train_data.drop(columns=[self.dataset.target_attribute])
        y_train = train_data[self.dataset.target_attribute]

        X_test = test_data.drop(columns=[self.dataset.target_attribute])
        y_test = test_data[self.dataset.target_attribute]

        automl = AutoML()

        if self.dataset.task_type == 'binary':
            task_type = "classification"
            pref_metric = "roc_auc"

        elif self.dataset.task_type == 'multiclass':
            task_type = "classification"
            pref_metric = "roc_auc_ovr"
        else:
            task_type = "regression"
            pref_metric = "r2"

        flaml_log_file_name = os.path.join(self.config.output_dir, "flaml.log")
        self.touch(path=self.config.output_dir, as_dir=True)

        automl.fit(X_train=X_train,
               y_train=y_train,
               metric=pref_metric,
               task=task_type,
               n_jobs=self.config.nthreads,
               log_file_name=flaml_log_file_name,
               time_budget=self.config.max_runtime_seconds)

        time_end = time.time()
        time_execute = time_end - time_start

        # Extract Results
        if self.dataset.task_type == "binary":
            y_train_pred = automl.predict(X_train)
            y_test_pred = automl.predict(X_test)

            self.log_results.train_accuracy = automl.score(X_train, y_train)
            self.log_results.train_f1_score = f1_score(y_train, y_train_pred, average='weighted')
            self.log_results.train_auc = roc_auc_score(y_train, automl.predict_proba(X_train)[:, 1])

            self.log_results.test_accuracy = automl.score(X_test, y_test)
            self.log_results.test_f1_score = f1_score(y_test, y_test_pred, average='weighted')
            self.log_results.test_auc = roc_auc_score(y_test, automl.predict_proba(X_test)[:, 1])

        elif self.dataset.task_type == "multiclass":
            predictions_train = automl.predict_proba(train_data)
            predictions_test = automl.predict_proba(test_data)

            self.log_results.train_accuracy = automl.score(X_train,y_train)
            self.log_results.train_auc_ovr = roc_auc_score(y_train, predictions_train, multi_class='ovr')
            self.log_results.train_auc_ovo = roc_auc_score(y_train, predictions_train, multi_class='ovo')
            self.log_results.train_log_loss = log_loss(y_train, predictions_train)

            self.log_results.test_accuracy = automl.score(X_test, y_test)
            self.log_results.test_auc_ovr = roc_auc_score(y_test, predictions_test, multi_class='ovr')
            self.log_results.test_auc_ovo = roc_auc_score(y_test, predictions_test, multi_class='ovo')
            self.log_results.test_log_loss = log_loss(y_test, predictions_test)

        elif self.dataset.task_type == "regression":
            y_train_pred = automl.predict(X_train)
            y_test_pred = automl.predict(X_test)

            self.log_results.train_r_squared = r2_score(y_train, y_train_pred)
            self.log_results.train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)

            self.log_results.test_r_squared = r2_score(y_test, y_test_pred)
            self.log_results.test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

        self.log_results.number_iteration = self.config.iteration
        self.log_results.llm_model = self.config.llm_model
        self.log_results.status = "True"
        self.log_results.time_execution = time_execute
        self.log_results.config = "Flaml"
        self.log_results.sub_task = self.config.sub_task
        self.log_results.save_results(result_output_path=self.config.output_path)



