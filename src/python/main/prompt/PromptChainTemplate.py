from .BasicPrompt import BasicPrompt
from util import StaticValues


class DataPreprocessingChainPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        from util.Config import _system_delimiter
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

        self.rules = [f'{_system_delimiter} {StaticValues.dp_rule_task}',
                      f'{_system_delimiter} {StaticValues.dp_rule_input}',
                      f'{_system_delimiter} {StaticValues.dp_rule_output}',
                      f'#1 : {StaticValues.dp_rule_1}',
                      f'#2 : {StaticValues.dp_rule_2.format(self.data_source_train_path, self.data_source_test_path)}',
                      f'#3 : {StaticValues.dp_rule_3}',
                      f'#4 : {StaticValues.dp_rule_4.format(self.ds_attribute_prefix, self.ds_attribute_prefix_label)}',
                      f'#5 : {StaticValues.dp_rule_5}',
                      f'#6 : {StaticValues.dp_rule_6}',
                      f'#7 : {StaticValues.dp_rule_7.format(self.target_attribute)}',
                      f'#8 : {StaticValues.dp_rule_8}',
                      f'#9 : {StaticValues.rule_code_block}',
                      f'#10 : {StaticValues.chain_rule_1}']

        self.question = "Provide a pipeline code that do data preprocessing in a multi-threaded environment."


class FeatureEngineeringChainPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        from util.Config import _system_delimiter
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

        self.rules = [f'{_system_delimiter} {StaticValues.fe_rule_task.format(self.target_attribute)}',
                      f'{_system_delimiter} {StaticValues.fe_rule_input}',
                      f'{_system_delimiter} {StaticValues.fe_rule_output}',
                      f'#1 : {StaticValues.fe_rule_1.format(self.target_attribute)}',
                      f'#2 : {StaticValues.fe_rule_2.format(algorithm, self.target_attribute)}',
                      f'#3 : {StaticValues.fe_rule_3}',
                      f'#4 : {StaticValues.fe_rule_4}',
                      f'#5 : {StaticValues.rule_code_block}',
                      f'#6 : {StaticValues.chain_rule_1}',
                      f'#7 : {StaticValues.dp_rule_2.format(self.data_source_train_path, self.data_source_test_path)}',]

        self.question = ("Provide a pipeline code that modify the Data Preprocessing code by adding Feature Engineering"
                         " tasks in a multi-threaded environment.")


class ModelSelectionChainPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        from util.Config import _system_delimiter
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

        randomforest_param = 500
        self.previous_result_format = f"<CODE>\n{self.previous_result}\n</CODE>"

        self.rules = [f'{_system_delimiter} {StaticValues.ms_rule_task.format(algorithm)}',
                      f'{_system_delimiter} {StaticValues.ms_rule_input}',
                      f'{_system_delimiter} {StaticValues.ms_rule_output}',
                      f'#1 : {StaticValues.ms_rule_1.format(algorithm, self.target_attribute)}',
                      f'#2 : {StaticValues.ms_rule_2.format(randomforest_param)}',
                      f'#3 : {self.evaluation_text}',
                      f'#4 : {StaticValues.ms_rule_3}',
                      f'#5 : {StaticValues.rule_code_block}',
                      f'#6 : {StaticValues.chain_rule_1}',
                      f'#7 : {StaticValues.dp_rule_2.format(self.data_source_train_path, self.data_source_test_path)}']

        self.question = ("Provide a complete pipeline code that can be executed in a multi-threaded environment.")