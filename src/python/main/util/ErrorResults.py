import pandas as pd
import time

class ErrorResults(object):
    def __init__(self,
                 error_class: str,
                 error_type: str,
                 error_value: str,
                 error_detail: str,
                 error_exception: str,
                 dataset_name: str,
                 llm_model: str,
                 config: str,
                 sub_task: str,
                 file_name: str,
                 timestamp
                 ):

        self.error_class = error_class
        self.error_type = error_type
        self.error_value = error_value
        self.error_detail = error_detail
        self.error_exception = error_exception
        self.dataset_name = dataset_name
        self.llm_model= llm_model
        self.config = config
        self.sub_task = sub_task
        self.file_name = file_name
        self.timestamp = timestamp

        self.columns = ["row_id", "dataset_name", "llm_model", "config", "sub_task", "error_class", "error_type",
                        "error_value", "error_detail", "error_exception", "file_name", "timestamp"]

    def save_error(self, error_output_path: str):
        try:
            df_error = pd.read_csv(error_output_path)

        except Exception as err:
            df_error = pd.DataFrame(columns=self.columns)

        row_id = len(df_error) + 1
        df_error.loc[len(df_error)] = [row_id,
                                       self.dataset_name,
                                       self.llm_model,
                                       self.config,
                                       self.sub_task,
                                       self.error_class,
                                       self.error_type,
                                       self.error_value,
                                       self.error_detail,
                                       self.error_exception,
                                       self.file_name,
                                       self.timestamp]

        df_error.to_csv(error_output_path, index=False)
