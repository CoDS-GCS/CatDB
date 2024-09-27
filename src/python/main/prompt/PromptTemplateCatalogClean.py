from .BasicPrompt import BasicPrompt


class CatalogCleaningPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.question = None
        self.data_cleaning_type = None
        self.df_content = None
        self.set_schema_content()

    def format_user_message(self):
        from util.Config import _user_delimiter
        prompt_items = []
        schema_data = self.format_schema_none_categorical_data()
        if schema_data is None:
            return None, None
        prompt_items.append(schema_data)
        prompt_items.append(f'Question: {self.question}')
        return f"\n\n{_user_delimiter}".join(prompt_items), None

    def format_schema_none_categorical_data(self):
        content = []
        for r in range(0, len(self.df_content)):
            row_msg = []
            if (self.df_content.loc[r]["column_data_type"] in {'str', 'date'} and self.df_content.loc[r]["is_categorical"] == False):
                row_msg.append(f'# Column Name is "{self.df_content.loc[r]["column_name"]}"')
                row_msg.append(f'# Distinct-Percentage [{self.df_content.loc[r]["distinct_count"]/self.catalog.nrows}%]')
                row_msg.append(f'samples [{self.df_content.loc[r]["samples"]}]')
                content.append(", ".join(row_msg))
        if len(content) == 0:
            return None
        content = "\n".join(content)
        prompt_items = [f'Column Names and Samples:',
                        '"""',
                        content,
                        '"""']
        return "\n".join(prompt_items)

    def format_system_message(self):
        from util.Config import _system_delimiter, _catdb_categorical_catalog_cleaning_rules as _catdb_rules
        rules = [f"{_system_delimiter} {_catdb_rules['task']}",
                 f"{_system_delimiter} {_catdb_rules['input']}",
                 f"{_system_delimiter} {_catdb_rules['output']}",
                 f"# 1: {_catdb_rules['Rule_1']}",
                 f"# 2: {_catdb_rules['Rule_2']}",
                 f"# 3: {_catdb_rules['Rule_3']}",
                 f"# 4: {_catdb_rules['Rule_4']}",
                 f"# 5: {_catdb_rules['Rule_5']}"
                 ]

        rule_msg = "\n".join(rules)
        return rule_msg


class CatDBCatalogCleanPrompt(CatalogCleaningPrompt):
    def __init__(self, *args, **kwargs):
        CatalogCleaningPrompt.__init__(self,
                                       flag_categorical_values=False,
                                       flag_missing_value_frequency=False,
                                       flag_dataset_description=False,
                                       flag_distinct_value_count=True,
                                       flag_statistical_number=False,
                                       flag_samples=True,
                                       flag_previous_result=False,
                                       *args, **kwargs)
        self.question = f'Infer that the mentioned columns are a categorical column or not.'
