import pandas as pd


class LogResults(object):
    def __init__(self,
                 dataset_name: str,
                 task_type: str,
                 status: str="False",
                 number_iteration: int=1,
                 cleaning_plan: str=None,
                 orig_feature_count: int=0,
                 clean_feature_count: int=0,
                 orig_samples:int=0,
                 clean_samples: int=0,
                 removed_features: str=None,
                 total_time: float = 0,
                 ):
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.status = status
        self.number_iteration = number_iteration
        self.cleaning_plan = cleaning_plan
        self.orig_feature_count = orig_feature_count
        self.clean_feature_count = clean_feature_count
        self.orig_samples = orig_samples
        self.clean_samples = clean_samples
        self.removed_features = removed_features
        self.total_time = total_time

        self.columns = ["dataset_name", "task_type", "status", "number_iteration", "cleaning_plan", "orig_feature_count",
                 "clean_feature_count", "orig_samples", "clean_samples", "removed_features", "total_time"]

    def save_results(self, result_output_path: str):
        try:
            df_result = pd.read_csv(result_output_path)

        except Exception as err:
            df_result = pd.DataFrame(columns=self.columns)

        df_result.loc[len(df_result)] = [self.dataset_name,
        self.task_type,
        self.status,
        self.number_iteration,
        self.cleaning_plan,
        self.orig_feature_count,
        self.clean_feature_count,
        self.orig_samples,
        self.clean_samples,
        self.removed_features,
        self.total_time,]

        df_result.to_csv(result_output_path, index=False)
