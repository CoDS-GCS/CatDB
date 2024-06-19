import os
import random


class Config(object):
    def __init__(self,
                 nthreads: int = os.cpu_count(),
                 jvm_memory: int = 0,
                 seed: int = random.randint(1, (1 << 31) - 1),
                 max_runtime_seconds: int = 0,
                 output_predictions_file_train: str = None,
                 output_predictions_file_test: str = None,
                 output_dir: str = None,
                 name: str = None,
                 output_path: str= None,
                 iteration: int= 1
                 ):
        self.nthreads = nthreads
        self.jvm_memory = jvm_memory
        self.seed = seed
        self.max_runtime_seconds = max_runtime_seconds
        self.output_predictions_file_train = output_predictions_file_train
        self.output_predictions_file_test = output_predictions_file_test
        self.output_dir = output_dir
        self.name = name
        self.output_path = output_path
        self.iteration = iteration

    def get_metrics(self, task_type: str):
        metrics = []
        if task_type == 'binary':
            metrics = ['logloss', 'acc', 'f1', 'auc', 'pr_auc']

        elif task_type == 'multiclass':
            metrics = ['logloss', 'acc', 'balacc', 'f1', 'auc_ovo', 'auc_ovr']

        elif task_type == 'regression':
            metrics = ['rmse', 'r2', 'mae', 'mse', 'rmsle']

        return metrics
