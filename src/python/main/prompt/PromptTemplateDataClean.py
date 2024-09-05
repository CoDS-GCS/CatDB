from .BasicPrompt import BasicPrompt
import pandas as pd


class CategoricalDataCleanPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.question = None

    def format_user_message(self):
        from util.Config import _user_delimiter, _DATASET_DESCRIPTION, _missing_value_train_data
        prompt_items = []
        if self.flag_dataset_description and self.dataset_description is not None:
            prompt_items.append(_user_delimiter+" "+_DATASET_DESCRIPTION.dataset_description.format(self.dataset_description))

        prompt_items.append(f"Table Data:\n{_missing_value_train_data}")
        schema_data = self.format_schema_data()
        prompt_items.append(schema_data)
        prompt_items.append(f'Question: {self.question}')
        return f"\n\n{_user_delimiter}".join(prompt_items), schema_data


    def format_user_message(self):
        from util.Config import _user_delimiter, _DATASET_DESCRIPTION
        prompt_items = []

        schema_data = []
        columns_categorical = []
        for cat in self.catalog:
            if cat.table_name == self.target_table:
                nrows = cat.nrows
            schema_data.append(self.format_schema_data(table_name= cat.table_name))
            if len(cat.columns_categorical) > 0:
                columns_categorical.extend(cat.columns_categorical)

        schema_data = "\n\n".join(schema_data)
        prompt_items.append(schema_data)

        # Relation Prompt:
        dependency_items = [f'There are following relationships between Tables. This relation define by Primary Keys and Forigin Keys.']
        for cat in self.catalog:
            if cat.dependency.primary_keys is not None:
                tmp_txt = "is"
                if len(cat.dependency.primary_keys ) > 1:
                    tmp_txt = "are"
                dependency_items.append(f"# {','.join(cat.dependency.primary_keys)} {tmp_txt} primary key in Table \"{cat.table_name}\".")

            if cat.dependency.foreign_keys is not None:
                tmp_txt = "is"
                if len(cat.dependency.foreign_keys ) > 1:
                    tmp_txt = "are"
                dependency_items.append(f"# {','.join(cat.dependency.foreign_keys)} {tmp_txt} foreign key in Table \"{cat.table_name}\".")
        dependency_items = "\n".join(dependency_items)
        prompt_items.append(dependency_items)

        if self.flag_missing_value_frequency:
            missing_values_rules = self.get_missing_values_rules()
            for k in missing_values_rules.keys():
                if missing_values_rules[k] is not None:
                    prompt_items.append(missing_values_rules[k])

        scaler_prompt = self.get_scaler_rules()
        if scaler_prompt is not None:
            prompt_items.append(scaler_prompt)
        # Encode categorical values:
        if self.flag_categorical_values and len(columns_categorical) > 0:
            categorical_columns = []
            for cc in columns_categorical:
                if cc != self.target_attribute:
                    categorical_columns.append(cc)
            categorical_column_prompt = (f'Transformer the categorical data for the following (e.g., One-Hot Encoding, Ordinal Encoder, Polynomial Encoder, Count Encoder, ... ) '
                                         f'columns:\n\t# Columns: {",".join(categorical_columns)}')
            prompt_items.append(categorical_column_prompt)

        prompt_items.append(f"Dataset Attribute:\n# Number of samples (rows) in training dataset: {nrows}")

        prompt_items.append(f'Dataset is a structured/tabular data, select a high performance ML model. For example, Gradient Boosting Machines (e.g., XGBoost, LightGBM, ...), RandomForest, ...')
        prompt_items.append(f'Question: {self.question}')
        return f"\n\n{_user_delimiter}".join(prompt_items), schema_data
    def format_system_message(self):
        from util.Config import _system_delimiter, _catdb_rules
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

    def set_schema_content(self):
        self.df_content = pd.DataFrame(columns=["column_name",
                                                "column_data_type",
                                                "distinct_count",
                                                "is_numerical",
                                                "min_value",
                                                "max_value",
                                                "median",
                                                "mean",
                                                "is_categorical",
                                                "categorical_values",
                                                "categorical_values_ratio",
                                                "samples"])
        dropped_columns_names = self.catalog.drop_schema_info.keys()
        update_missed_columns = []
        for k in self.catalog.schema_info.keys():
            if k in dropped_columns_names or k not in self.missed_columns:
                continue
            update_missed_columns.append(k)
            cp = self.catalog.profile_info[k]

            is_numerical = False
            is_categorical = False
            categorical_values = None
            categorical_values_ratio = None
            samples_text = None

            if k in self.catalog.columns_numerical:
                is_numerical = True

            if cp.categorical_values is not None and k in self.catalog.columns_categorical:
                is_categorical = True
                if len(cp.categorical_values) > 10:
                    categorical_values = [str(val) for val in cp.categorical_values[0: 10]]
                    categorical_values.append(f"and {len(cp.categorical_values) - 10} more")
                else:
                    categorical_values = [str(val) for val in cp.categorical_values]

                categorical_values = (",".join(categorical_values)).replace("\"","'")
                tmp_cc = []
                # for cv in cp.categorical_values:
                #     tmp_cc.append(f"{cv}:{cp.categorical_values_ratio[str(cv)]}")

                categorical_values_ratio = (",".join(tmp_cc)).replace("\"","\'")

            if cp.samples is not None and len(cp.samples) > 0:
                samples_text = ",".join([str(val) for val in cp.samples[0:self.number_samples]])

            self.df_content.loc[len(self.df_content)] = [k, cp.short_data_type, cp.distinct_values_count, is_numerical,
                                                         cp.min_value, cp.max_value, cp.median, cp.mean, is_categorical,
                                                         categorical_values, categorical_values_ratio, samples_text]
        self.missed_columns = update_missed_columns


class CatDBCategoricalDataCleanPrompt(CategoricalDataCleanPrompt):
    def __init__(self, *args, **kwargs):
        CategoricalDataCleanPrompt.__init__(self,
                             flag_categorical_values=True,
                             flag_missing_value_frequency=False,
                             flag_dataset_description=True,
                             flag_distinct_value_count=True,
                             flag_statistical_number=True,
                             flag_samples=False,
                             flag_previous_result=False,
                             *args, **kwargs)
        self.ds_attribute_prefix = "Schema, and Data Profiling Info"
        self.ds_attribute_prefix_label = "Schema, and Data Profiling Info:"
        self.config = "CatDB"
        self.question = f'Predict missed values of {self.missed_columns}, and return scaler value(s) for the following {self.target_samples_size} samples:\n {self.target_samples}'