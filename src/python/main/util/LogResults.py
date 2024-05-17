import pandas as pd


class LogResults(object):
    def __init__(self,
                 dataset_name: str,
                 config: str,
                 sub_task: str,
                 llm_model: str,
                 classifier: str,
                 task_type: str,
                 status: str,
                 number_iteration: int,
                 number_iteration_error: int,
                 has_description: str,
                 time_catalog_load: float,
                 time_pipeline_generate: float,
                 time_total: float,
                 time_execution: float,
                 train_accuracy: float = -2,
                 train_f1_score: float = -2,
                 train_log_loss: float = -2,
                 train_r_squared: float = -2,
                 train_rmse: float = -2,
                 test_accuracy: float = -2,
                 test_f1_score: float = -2,
                 test_log_loss: float = -2,
                 test_r_squared: float = -2,
                 test_rmse: float = -2
                 ):
        self.config = config
        self.sub_task = sub_task
        self.llm_model = llm_model
        self.classifier = classifier
        self.task_type = task_type
        self.status = status
        self.number_iteration = number_iteration
        self.number_iteration_error = number_iteration_error
        self.has_description = has_description
        self.dataset_name = dataset_name
        self.time_catalog_load = time_catalog_load
        self.time_pipeline_generate = time_pipeline_generate
        self.time_total = time_total
        self.time_execution = time_execution
        self.train_accuracy = train_accuracy
        self.train_f1_score = train_f1_score
        self.train_log_loss = train_log_loss
        self.train_r_squared = train_r_squared
        self.train_rmse = train_rmse
        self.test_accuracy = test_accuracy
        self.test_f1_score = test_f1_score
        self.test_log_loss = test_log_loss
        self.test_r_squared = test_r_squared
        self.test_rmse = test_rmse

        self.columns = ["dataset_name", "config", "sub_task", "llm_model", "classifier", "task_type", "status",
                        "number_iteration","number_iteration_error", "has_description", "time_catalog_load", "time_pipeline_generate",
                        "time_total", "time_execution", "train_accuracy", "train_f1_score", "train_log_loss",
                        "train_r_squared", "train_rmse", "test_accuracy", "test_f1_score", "test_log_loss",
                        "test_r_squared", "test_rmse"]

    def save_results(self, result_output_path: str):
        try:
            df_result = pd.read_csv(result_output_path)

        except Exception as err:
            df_result = pd.DataFrame(columns=self.columns)

        df_result.loc[len(df_result)] = [self.dataset_name,
                                         self.config,
                                         self.sub_task,
                                         self.llm_model,
                                         self.classifier,
                                         self.task_type,
                                         self.status,
                                         self.number_iteration,
                                         self.number_iteration_error,
                                         self.has_description,
                                         self.time_catalog_load,
                                         self.time_pipeline_generate,
                                         self.time_total,
                                         self.time_execution,
                                         self.train_accuracy,
                                         self.train_f1_score,
                                         self.train_log_loss,
                                         self.train_r_squared,
                                         self.train_rmse,
                                         self.test_accuracy,
                                         self.test_f1_score,
                                         self.test_log_loss,
                                         self.test_r_squared,
                                         self.test_rmse]

        df_result.to_csv(result_output_path, index=False)
