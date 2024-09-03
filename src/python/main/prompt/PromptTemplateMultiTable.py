from .BasicPromptMultiTable import BasicPromptMultiTable


class CatDBMultiTablePrompt(BasicPromptMultiTable):
    def __init__(self, *args, **kwargs):
        BasicPromptMultiTable.__init__(self,
                             flag_categorical_values=True,
                             flag_missing_value_frequency=True,
                             flag_dataset_description=True,
                             flag_distinct_value_count=True,
                             flag_statistical_number=True,
                             flag_samples=True,
                             flag_previous_result=False,
                             *args, **kwargs)

        self.ds_attribute_prefix = "Schema, and Data Profiling Info"
        self.ds_attribute_prefix_label = "Schema, and Data Profiling Info "
        self.question = "Provide a complete pipeline code that can be executed in a multi-threaded environment."

    def format_system_message(self):
        from util.Config import (_system_delimiter, _catdb_rules, _CODE_FORMATTING_IMPORT, _CODE_FORMATTING_ADDING,
                                 _CODE_FORMATTING_DROPPING, _CODE_FORMATTING_TECHNIQUE, _CODE_BLOCK)
        self.schema_keys = [_ for _ in self.catalog[0].schema_info.keys()]
        if self.task_type == "binary classification" or self.task_type == "multiclass classification":
            algorithm = "classifier"
        else:
            algorithm = "regressor"

        tables = []
        for cat in self.catalog:
            tables.append(cat.table_name)

        rules = [f"{_system_delimiter} {_catdb_rules['task'].format(self.ds_attribute_prefix)}",
                     f"{_system_delimiter} {_catdb_rules['input'].format(','.join(tables))}",
                     f"{_system_delimiter} {_catdb_rules['output']}",
                     f"# 1: {_catdb_rules['Rule_1']}",
                     f"# 2: {_catdb_rules['Rule_2'].format(self.data_source_train_path, self.data_source_test_path)}",
                     f"# 3: {_catdb_rules['Rule_3']}",
                     f"# 4: {_catdb_rules['Rule_4'].format(self.ds_attribute_prefix, self.ds_attribute_prefix_label)}",
                     f"# 5: {_catdb_rules['Rule_5']}",
                     f"# 6: {_catdb_rules['Rule_6']}",
                     f"# 7: {_catdb_rules['Rule_7']}",
                     f"# 8: {_catdb_rules['Rule_8']}",
                     f"# 9: {_catdb_rules['Rule_9']}",
                     f"# 10: {_catdb_rules['Rule_10'].format(self.target_attribute)}",
                     f"# 11: {_catdb_rules['Rule_11'].format(algorithm, self.target_attribute)}",
                     f"# 12: {_catdb_rules['Rule_12']}",
                     f"# 13: {_CODE_FORMATTING_IMPORT}",
                     f"# 14: {_CODE_FORMATTING_ADDING.format(self.target_attribute, self.schema_keys[0], self.schema_keys[1])}",
                     f"# 15: {_CODE_FORMATTING_DROPPING}",
                     f"# 16: {_CODE_FORMATTING_TECHNIQUE.format(algorithm)}",
                     f"# 17: {self.evaluation_text}",
                     f"# 18: {_catdb_rules['Rule_13']}",
                     f"# 19: {_catdb_rules['Rule_14']}",
                     f"# 20: {_CODE_BLOCK}"
                     ]
        rule_msg = "\n".join(rules)
        return rule_msg