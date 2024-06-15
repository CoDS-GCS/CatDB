from automl.AutoML import AutoML as CatDBAutoML
from util.Config import Config
from util.Data import Dataset, reader_CSV

import os
from flaml import AutoML, __version__


class FlamlAutoML(CatDBAutoML):
    def __init__(self, dataset: Dataset, config: Config, *args, **kwargs):
        CatDBAutoML.__init__(self, dataset=dataset, config=config)

    def run(self):
        print(f"\n**** FLAML [v{__version__}] ****\n")

        train_data = reader_CSV(self.dataset.train_path)
        test_data = reader_CSV(self.dataset.test_path)
        X_train = train_data.drop(columns=[self.dataset.target_attribute])
        y_train = train_data[self.dataset.target_attribute].squeeze()

        is_classification = self.dataset.task_type == 'classification'
        time_budget = self.config.max_runtime_seconds
        n_jobs = self.config.nthreads
        ml = AutoML()

        # Mapping of benchmark metrics to flaml metrics
        metrics_mapping = dict(
            acc='accuracy',
            auc='roc_auc',
            f1='f1',
            logloss='log_loss',
            mae='mae',
            mse='mse',
            rmse='rmse',
            r2='r2',
        )
        metrics = self.config.get_metrics(self.dataset.task_type)
        sort_metric = []
        for m in metrics:
            metric = metrics_mapping.get(m)
            if metric is not None:
                sort_metric.append(metric)

        flaml_log_file_name = os.path.join(self.config.output_dir, "flaml.log")
        self.touch(path=self.config.output_dir, as_dir=True)

        ml.fit(X_train= X_train,
               y_train= y_train,
               metric="accuracy",
               task='classification',#self.dataset.task_type,
               n_jobs=n_jobs,
               log_file_name=flaml_log_file_name,
               time_budget=time_budget)
