import pandas as pd
from catalog.Dependency import Dependency

class BasicPromptMultiTable(object):
    def __init__(self,
                 flag_distinct_value_count: bool = False,
                 flag_missing_value_frequency: bool = False,
                 flag_statistical_number: bool = False,
                 flag_categorical_values: bool = False,
                 flag_dataset_description: bool = False,
                 flag_previous_result: bool = False,
                 flag_samples: bool = False,
                 *args, **kwargs):
        self.ds_attribute_prefix = None
        self.ds_attribute_prefix_label = None
        self.rules = []
        self.schema_keys = None
        self.question = None
        self.extra_info = None
        self.df_content = dict()
        self.set_schema_content()

        self.flag_distinct_value_count = flag_distinct_value_count
        self.flag_missing_value_frequency = flag_missing_value_frequency
        self.flag_statistical_number = flag_statistical_number
        self.flag_categorical_values = flag_categorical_values
        self.flag_dataset_description = flag_dataset_description
        self.flag_previous_result = flag_previous_result
        self.flag_samples = flag_samples
        self.previous_result_format = None

    def format(self):
        user_message, schema_data = self.format_user_message()
        return {
            "system_message": self.format_system_message(),
            "user_message": user_message,
            "schema_data": schema_data
        }

    def format_user_message(self):
        from util.Config import _user_delimiter, _DATASET_DESCRIPTION
        prompt_items = []
        if self.flag_dataset_description and self.dataset_description is not None:
            prompt_items.append(_user_delimiter+" "+_DATASET_DESCRIPTION.dataset_description.format(self.dataset_description))

        if self.flag_previous_result and self.previous_result is not None:
            prompt_items.append(self.previous_result_format)

        nrows = 0
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
        return "\n".join(self.rules)

    def floatToString(self, inputValue):
        return ('%.3f' % inputValue).rstrip('0').rstrip('.')

    def format_schema_data(self, table_name):
        content = []
        df_content = self.df_content[table_name]
        target_text = "**This is a target column**"
        for r in range(0, len(df_content)):
            if df_content.loc[r]["column_name"] == self.target_attribute:
                row_msg_1 = f'# \"{df_content.loc[r]["column_name"]}\" ({df_content.loc[r]["column_data_type"]}, {target_text})'
            else:
                row_msg_1 = f'# {df_content.loc[r]["column_name"]} ({df_content.loc[r]["column_data_type"]})'
            row_msg = [row_msg_1]
            if self.flag_distinct_value_count and df_content.loc[r]["is_categorical"] == False:
                row_msg.append(f'distinct-count [{df_content.loc[r]["distinct_count"]}]')

            if df_content.loc[r]["is_numerical"] and self.flag_statistical_number:
                row_msg.append(f'min-value [{self.floatToString(df_content.loc[r]["min_value"])}]')
                row_msg.append(f'max-value [{self.floatToString(df_content.loc[r]["max_value"])}]')
                row_msg.append(f'median-value [{self.floatToString(df_content.loc[r]["median"])}]')
                row_msg.append(f'mean-value [{self.floatToString(df_content.loc[r]["mean"])}]')

            if df_content.loc[r]["is_categorical"] and self.flag_categorical_values:
                row_msg.append(f'categorical-values [{df_content.loc[r]["categorical_values"]}]')

            elif self.number_samples > 0 and self.flag_samples:
                row_msg.append(f'samples [{df_content.loc[r]["samples"]}]')

            content.append(", ".join(row_msg))
        content = "\n".join(content)
        prompt_items = [f'{self.ds_attribute_prefix_label} Table """{table_name}""":',
                        '"""',
                        content,
                        '"""']
        return "\n".join(prompt_items)

    def get_missing_values_rules(self):
        missing_values_rules = {"numerical_missing_values": None,
                                "bool_missing_values": None,
                                "categorical_missing_values": None,
                                "others_missing_values": None}

        columns_numerical_missing_values = []
        columns_bool_missing_values = []
        columns_categorical_missing_values = []
        columns_others_missing_values = []

        for cat in self.catalog:
            if len(cat.columns_numerical_missing_values) > 0:
                columns_numerical_missing_values.extend(cat.columns_bool_missing_values)

            if len(cat.columns_bool_missing_values) > 0:
                    columns_bool_missing_values.extend(cat.columns_bool_missing_values)

            if len(cat.columns_categorical_missing_values) > 0:
                    columns_categorical_missing_values.extend(cat.columns_categorical_missing_values)

            if len(cat.columns_others_missing_values) > 0:
                columns_others_missing_values.extend(cat.columns_others_missing_values)

        if len(columns_numerical_missing_values) > 0:
            missing_values_prompt = (f"Do missing values imputation semantically for the following numerical columns:\n\tColumns: "
                                     f"{','.join(columns_numerical_missing_values)}\n")
            missing_values_rules["numerical_missing_values"] = missing_values_prompt

        if len(columns_bool_missing_values) > 0:
            missing_values_prompt = (
                f"Predict the missing values semantically for the following boolean columns:\n\tColumns: "
                f"{','.join(columns_bool_missing_values)}")
            missing_values_rules["bool_missing_values"] = missing_values_prompt

        if len(columns_categorical_missing_values) > 0:
            missing_values_prompt = (f"Predict the missing values semantically for the following categorical columns:\n\tColumns: "
                                     f"{','.join(columns_categorical_missing_values)}\n")
            missing_values_rules["categorical_missing_values"] = missing_values_prompt

        if len(columns_others_missing_values) > 0:
            missing_values_prompt = (
                f"Predict the missing values semantically for the following string/object columns:\n\tColumns: "
                f"{','.join(columns_others_missing_values)}")
            missing_values_rules["others_missing_values"] = missing_values_prompt

        return missing_values_rules

    def get_scaler_rules(self):
        if self.number_samples > 0:
            return self.get_scaler_rules_few_shot()
        else:
            return self.get_scaler_rules_zero_shot()

    def get_scaler_rules_zero_shot(self):
        transfer_columns = []
        for cat in self.catalog:
            for cc in cat.columns_numerical:
                if cc in cat.columns_categorical or cc in cat.columns_categorical:
                    continue
                else:
                    transfer_columns.append(cc)

        transfer_column_prompt = (
            f'Transformer the following columns by Adaptive Binning or Scaler method (do it base on the min-max, mean, and median values are in the "Schema, and Data Profiling Info"):\n '
            f'\t# Columns: {",".join(transfer_columns)}')
        if len(transfer_columns) > 0:
            return transfer_column_prompt
        else:
            return None

    def get_scaler_rules_few_shot(self):
        columns_numerical_missing_values = []
        for cat in self.catalog:
            if len(cat.columns_numerical_missing_values) > 0:
                columns_numerical_missing_values.extend(cat.drop_schema_info.keys())

        if len(columns_numerical_missing_values) > 0:
            scaler_prompt = (f"Select an appropriate scaler the following numerical columns "
                             f'(do it base on the min-max, mean, median, and samples values are in the '
                             f'"Schema, and Data Profiling Info"):\n\t'
                             f"Columns: {','.join(columns_numerical_missing_values)}\n")
            return scaler_prompt
        else:
            return None

    def get_drop_columns_rules(self):
        dropped_columns_names = []
        for cat in self.catalog:
            dropped_columns_names.extend(cat.drop_schema_info.keys())

        if len(dropped_columns_names) > 0:
            drop_column_prompt = "Drop the following column(s) from the train and test datasets:\n\tColumn(s): "
            names = []
            for k in dropped_columns_names:
                names.append(k)
            return f"{drop_column_prompt}{','.join(names)}\n"
        else:
            return None

    def set_schema_content(self):
        for cat in self.catalog:
            df_content = pd.DataFrame(columns=["column_name",
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
            dropped_columns_names = cat.drop_schema_info.keys()
            for k in cat.schema_info.keys():
                if k in dropped_columns_names:
                    continue

                cp = cat.profile_info[k]

                if cp.short_data_type == 'list':
                    continue

                is_numerical = False
                is_categorical = False
                categorical_values = None
                categorical_values_ratio = None
                samples_text = None

                if k in cat.columns_numerical:
                    is_numerical = True

                if cp.categorical_values is not None and k in cat.columns_categorical:
                    is_categorical = True
                    if len(cp.categorical_values) > 10:
                        categorical_values = [str(val) for val in cp.categorical_values[0: 10]]
                        categorical_values.append(f"and {len(cp.categorical_values) - 10} more")
                    else:
                        categorical_values = [str(val) for val in cp.categorical_values]

                    categorical_values = (",".join(categorical_values)).replace("\"","'")
                    tmp_cc = []
                    categorical_values_ratio = (",".join(tmp_cc)).replace("\"","\'")

                if cp.samples is not None and len(cp.samples) > 0:
                    samples_text = ",".join([str(val) for val in cp.samples[0:self.number_samples]])

                df_content.loc[len(df_content)] = [k, cp.short_data_type, cp.distinct_values_count, is_numerical,
                                                             cp.min_value, cp.max_value, cp.median, cp.mean, is_categorical,
                                                             categorical_values, categorical_values_ratio, samples_text]

                self.df_content[cat.table_name] = df_content
