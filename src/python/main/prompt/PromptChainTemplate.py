from .BasicPrompt import BasicPrompt


class DataPreprocessingChainPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        from util.Config import _system_delimiter, _catdb_chain_DP_rules, _CODE_BLOCK, _CHAIN_RULE
        BasicPrompt.__init__(self,
                             flag_categorical_values=True,
                             flag_missing_value_frequency=True,
                             flag_dataset_description=True,
                             flag_distinct_value_count=True,
                             flag_statistical_number=True,
                             flag_samples=True,
                             flag_previous_result=False,
                             *args, **kwargs)
        self.ds_attribute_prefix = "Schema, and Data Profiling Info"
        self.ds_attribute_prefix_label = "Schema, and Data Profiling Info:"

        self.rules = [f'{_system_delimiter} {_catdb_chain_DP_rules["task"]}',
                      f'{_system_delimiter} {_catdb_chain_DP_rules["input"]}',
                      f'{_system_delimiter} {_catdb_chain_DP_rules["output"]}',
                      f'#1 : {_catdb_chain_DP_rules["Rule_1"]}',
                      f'#2 : {_catdb_chain_DP_rules["Rule_2"].format(self.data_source_train_path, self.data_source_test_path)}',
                      f'#3 : {_catdb_chain_DP_rules["Rule_3"]}',
                      f'#4 : {_catdb_chain_DP_rules["Rule_4"].format(self.ds_attribute_prefix, self.ds_attribute_prefix_label)}',
                      f'#5 : {_catdb_chain_DP_rules["Rule_5"]}',
                      f'#6 : {_catdb_chain_DP_rules["Rule_6"]}',
                      f'#7 : {_catdb_chain_DP_rules["Rule_7"]}',
                      f'#8 : {_catdb_chain_DP_rules["Rule_8"].format(self.target_attribute)}',
                      f'#9 : {_catdb_chain_DP_rules["Rule_9"]}',
                      f'#10 : {_catdb_chain_DP_rules["Rule_10"]}',
                      f'#11 : {_CODE_BLOCK}',
                      f'#12 : {_CHAIN_RULE}']
        if self.previous_result is not None:
            self.previous_result_format = f"<CODE>\n{self.previous_result}\n</CODE>"
            self.question = "Provide a pipeline code hat modify the Data Preprocessing code of privious chunk in a multi-threaded environment."
        else:
            self.question = "Provide a pipeline code that do data preprocessing in a multi-threaded environment."


class FeatureEngineeringChainPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        from util.Config import _system_delimiter, _catdb_chain_FE_rules, _CODE_BLOCK, _CHAIN_RULE
        BasicPrompt.__init__(self,
                             flag_categorical_values=True,
                             flag_missing_value_frequency=False,
                             flag_dataset_description=True,
                             flag_distinct_value_count=True,
                             flag_statistical_number=True,
                             flag_samples=True,
                             flag_previous_result=True,
                             *args, **kwargs)
        self.ds_attribute_prefix = "Schema, and Data Profiling Info"
        self.ds_attribute_prefix_label = "Schema, and Data Profiling Info:"

        if self.task_type == "binary classification" or self.task_type == "multiclass classification":
            algorithm = "classifier"
        else:
            algorithm = "regressor"

        self.previous_result_format = f"<CODE>\n{self.previous_result}\n</CODE>"

        self.rules = [f'{_system_delimiter} {_catdb_chain_FE_rules["task"].format(self.target_attribute)}',
                      f'{_system_delimiter} {_catdb_chain_FE_rules["input"]}',
                      f'{_system_delimiter} {_catdb_chain_FE_rules["output"]}',
                      f'#1 : {_catdb_chain_FE_rules["Rule_1"].format(self.data_source_train_path, self.data_source_test_path)}',
                      f'#2 : {_catdb_chain_FE_rules["Rule_2"]}',
                      f'#3 : {_catdb_chain_FE_rules["Rule_3"].format(self.target_attribute)}',
                      f'#4 : {_catdb_chain_FE_rules["Rule_4"].format(algorithm, self.target_attribute)}',
                      f'#5 : {_catdb_chain_FE_rules["Rule_5"]}',
                      f'#6 : {_catdb_chain_FE_rules["Rule_6"]}',
                      f'#7 : {_CODE_BLOCK}',
                      f'#8 : {_CHAIN_RULE}']

        self.question = ("Provide a pipeline code that modify the Data Preprocessing code by adding Feature Engineering"
                         " tasks in a multi-threaded environment.")


class ModelSelectionChainPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        from util.Config import _system_delimiter, _catdb_chain_MS_rules, _CODE_BLOCK, _CHAIN_RULE
        BasicPrompt.__init__(self,
                             flag_categorical_values=True,
                             flag_missing_value_frequency=False,
                             flag_dataset_description=True,
                             flag_distinct_value_count=True,
                             flag_statistical_number=False,
                             flag_samples=True,
                             flag_previous_result=True,
                             *args, **kwargs)
        self.ds_attribute_prefix = "Schema, and Data Profiling Info"
        self.ds_attribute_prefix_label = "Schema, and Data Profiling Info:"

        if self.task_type == "binary classification" or self.task_type == "multiclass classification":
            algorithm = "classifier"
        else:
            algorithm = "regressor"

        self.previous_result_format = f"<CODE>\n{self.previous_result}\n</CODE>"

        self.rules = [f'{_system_delimiter} {_catdb_chain_MS_rules["task"].format(algorithm)}',
                      f'{_system_delimiter} {_catdb_chain_MS_rules["input"]}',
                      f'{_system_delimiter} {_catdb_chain_MS_rules["output"]}',
                      f'#1 : {_catdb_chain_MS_rules["Rule_1"].format(self.data_source_train_path, self.data_source_test_path)}',
                      f'#2 : {_catdb_chain_MS_rules["Rule_2"].format(algorithm, self.target_attribute)}',
                      f'#3 : {self.evaluation_text}',
                      f'#4 : {_catdb_chain_MS_rules["Rule_3"]}',
                      f'#5 : {_catdb_chain_MS_rules["Rule_4"]}'
                      f'#6 : {_CODE_BLOCK}',
                      f'#7 : {_CHAIN_RULE}']

        self.question = ("Provide a complete pipeline code that can be executed in a multi-threaded environment.")