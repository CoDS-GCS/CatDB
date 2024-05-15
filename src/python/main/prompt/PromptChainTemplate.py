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
                             *args, **kwargs)
        self.ds_attribute_prefix = "Schema, and Data Profiling Info"
        self.ds_attribute_prefix_label = "Schema, and Data Profiling Info:"

        self.rules = [f'{_system_delimiter} {StaticValues.dp_rule_task}',
                      f'{_system_delimiter} {StaticValues.Rule_input}',
                      f'{_system_delimiter} {StaticValues.Rule_output}',
                      f'#1 : {StaticValues.dp_rule_1}',
                      f'#2 : {StaticValues.dp_rule_2.format(self.data_source_train_path, self.data_source_test_path)}',
                      f'#3 : {StaticValues.dp_rule_3}',
                      f'#4 : {StaticValues.dp_rule_4.format(self.ds_attribute_prefix, self.ds_attribute_prefix_label)}',
                      f'#5 : {StaticValues.dp_rule_5}',
                      f'#6 : {StaticValues.dp_rule_6}',
                      f'#7 : {StaticValues.dp_rule_7.format(self.target_attribute)}']

        self.question = ("Provide a pipeline code that do data preprocessing in a multi-threaded environment "
                         "Each codeblock ends with \"```end\" and starts with \"```python\".")
