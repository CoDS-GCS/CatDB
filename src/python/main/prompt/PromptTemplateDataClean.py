import math

from .BasicPrompt import BasicPrompt


class DataCleaningPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.question = None
        self.data_cleaning_type = None
        self.df_content = None
        self.set_schema_content()
        self.parts = self.get_prompt_parts()

    def get_parts(self):
        return self.parts

    def get_prompt_parts(self):
        text_size = 0
        col_vals = dict()
        col_vals_len = dict()
        for r in range(0, len(self.df_content)):
            if self.df_content.loc[r]["is_categorical"] and self.flag_categorical_values and (
                    self.df_content.loc[r]["column_data_type"] == 'str' and
                    self.df_content.loc[r]["categorical_values_count"] > 2):
                text_size += len(f'# Column Name is "{self.df_content.loc[r]["column_name"]}"')
                text_size += len(f'categorical-values [{self.df_content.loc[r]["categorical_values"]}]')
                col_name = self.df_content.loc[r]["column_name"]
                col_val = self.column_categorical_vals[col_name]
                col_vals[col_name] = col_val
                col_vals_len[col_name] = len(self.df_content.loc[r]["categorical_values"])

        col_vals_len = {k: v for k, v in sorted(col_vals_len.items(), key=lambda item: item[1])}
        max_part_size = 4 * 4500
        part_size = 0
        part = dict()
        parts = []
        for c in col_vals_len.keys():
            if col_vals_len[c] + len(c) + part_size <= max_part_size:
                part[c] = col_vals[c]
                part_size += col_vals_len[c] + len(c)

            else:
                remain_len = max_part_size - part_size
                last_item = 0
                if remain_len > 1000:
                    part_size += len(c)
                    chunk_vals = []
                    for v in col_vals[c]:
                        if part_size + len(v) <= max_part_size:
                            chunk_vals.append(v)
                            last_item += 1
                            part_size += len(v)
                        else:
                            break
                    part[c] = chunk_vals
                parts.append(part)
                part_size = 0
                part = dict()
                while last_item < len(col_vals[c]):
                    chunk_vals = []
                    part_size += len(c)
                    for i in range(last_item, len(col_vals[c])):
                        if part_size + len(col_vals[c][i]) <= max_part_size:
                            chunk_vals.append(col_vals[c][i])
                            last_item += 1
                            part_size += len(col_vals[c][i])
                        else:
                            part[c] = chunk_vals
                            parts.append(part)
                            part_size = 0
                            part = dict()
                            break

        if len(parts) == 0:
            parts.append(part)
        return parts

    def format(self, part_id: int = -1):
        user_message, schema_data = self.format_user_message_multi_part(part_id)
        return {
            "system_message": self.format_system_message(),
            "user_message": user_message,
            "schema_data": schema_data
        }

    def format_user_message_multi_part(self, part_id):
        if self.data_cleaning_type == 'CategoricalData':
            return self.format_user_message_categorical_data_clean(part_id), None
        else:
            return None

    def format_user_message_categorical_data_clean(self, part_id):
        from util.Config import _user_delimiter
        prompt_items = []
        schema_data = self.format_schema_categorical_data(part_id)
        prompt_items.append(schema_data)
        prompt_items.append(f'Question: {self.question}')
        return f"\n\n{_user_delimiter}".join(prompt_items)

    def format_schema_categorical_data(self, part_id):
        content = []

        for c in self.parts[part_id]:
            row_msg = []
            categorical_values = [f'"{str(val)}"' for val in self.parts[part_id][c]]
            categorical_values = (",".join(categorical_values))
            row_msg.append(f'# Column Name is "{c}"')
            row_msg.append(f'categorical-values [{categorical_values}]')
            content.append(", ".join(row_msg))

        content = "\n".join(content)
        prompt_items = [f'Schema, and Categorical Data:',
                        '"""',
                        content,
                        '"""']
        return "\n".join(prompt_items)

    def format_system_message(self):
        from util.Config import _CODE_FORMATTING_IMPORT, _CODE_BLOCK
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
                 f"# 11: {_catdb_rules['Rule_11']}",
                 f"# 12: {_catdb_rules['Rule_12']}",
                 f"# 13: {_catdb_rules['Rule_13']}",
                 f"# 14: {_catdb_rules['Rule_14']}",
                 f"# 15: {_catdb_rules['Rule_15']}",
                 f"# 16: {_CODE_FORMATTING_IMPORT}",
                 f"# 17: {_CODE_BLOCK}"
                 ]

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
        self.question = f'Find the categorical data duplication and apply it on the dataframe.'
