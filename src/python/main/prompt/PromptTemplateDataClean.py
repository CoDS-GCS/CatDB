from .BasicPrompt import BasicPrompt
import pandas as pd


class DataCleaningPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.question = None
        self.data_cleaning_type = None
        self.df_content = None
        self.set_schema_content()

    def format_user_message(self):
        if self.data_cleaning_type == 'CategoricalData':
            return self.format_user_message_categorical_data_clean(), None
        else:
            return None

    def format_user_message_categorical_data_clean(self):
        from util.Config import _user_delimiter
        prompt_items = []

        schema_data = self.format_schema_categorical_data()
        #schema_data = "\n\n".join(schema_data)
        prompt_items.append(schema_data)
        prompt_items.append(f'Question: {self.question}')
        return f"\n\n{_user_delimiter}".join(prompt_items)

    def format_schema_categorical_data(self):
        content = []
        for r in range(0, len(self.df_content)):
            row_msg = []
            if self.df_content.loc[r]["is_categorical"] and self.flag_categorical_values and (self.df_content.loc[r]["column_data_type"] == 'str' and self.df_content.loc[r]["categorical_values_count"] > 2):
                row_msg.append(f'# Column Name is "{self.df_content.loc[r]["column_name"]}"')
                row_msg.append(f'categorical-values [{self.df_content.loc[r]["categorical_values"]}]')
                content.append(", ".join(row_msg))
        content = "\n".join(content)
        prompt_items = [f'Schema, and Categorical Data:',
                        '"""',
                        content,
                        '"""']
        return "\n".join(prompt_items)

    def format_system_message(self):
        from util.Config import _system_delimiter, _catdb_categorical_data_cleaning_rules as _catdb_rules
        rules = [f"{_system_delimiter} {_catdb_rules['task']}",
                 f"{_system_delimiter} {_catdb_rules['input']}",
                 f"{_system_delimiter} {_catdb_rules['output']}",
                 f"# 1: {_catdb_rules['Rule_1']}",
                 f"# 2: {_catdb_rules['Rule_2']}",
                 f"# 3: {_catdb_rules['Rule_3']}",
                 f"# 4: {_catdb_rules['Rule_4']}",
                 f"# 5: {_catdb_rules['Rule_5']}",
                 f"# 6: {_catdb_rules['Rule_6']}",
                 f"# 7: {_catdb_rules['Rule_7']}",
                 f"# 8: {_catdb_rules['Rule_8']}",
                 f"# 9: {_catdb_rules['Rule_9']}",
                 f"# 10: {_catdb_rules['Rule_10']}",
                 f"# 11: {_catdb_rules['Rule_11']}"]

        rule_msg = "\n".join(rules)
        return rule_msg


class CatDBCategoricalDataCleanPrompt(DataCleaningPrompt):
    def __init__(self, *args, **kwargs):
        DataCleaningPrompt.__init__(self,
                             flag_categorical_values=True,
                             flag_missing_value_frequency=False,
                             flag_dataset_description=True,
                             flag_distinct_value_count=True,
                             flag_statistical_number=True,
                             flag_samples=False,
                             flag_previous_result=False,
                             *args, **kwargs)
        self.data_cleaning_type = 'CategoricalData'
        self.question = f'Find the categorical data duplication and return a refined values.'