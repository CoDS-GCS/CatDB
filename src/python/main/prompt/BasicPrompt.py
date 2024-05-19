from util import StaticValues
import pandas as pd


class BasicPrompt(object):
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
        self.df_content = None
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
        return {
            "system_message": self.format_system_message(),
            "user_message": self.format_user_message()
        }

    def format_user_message(self):
        from util.Config import _user_delimiter
        prompt_items = []
        if self.flag_dataset_description and self.dataset_description is not None:
            prompt_items.append(_user_delimiter+" "+StaticValues.dataset_description.format(self.dataset_description))

        if self.flag_previous_result and self.previous_result is not None:
            prompt_items.append(self.previous_result_format)
        prompt_items.append(self.format_schema_data())

        if self.flag_missing_value_frequency:
            missing_values_rules = self.get_missing_values_rules()
            for k in missing_values_rules.keys():
                if missing_values_rules[k] is not None:
                    prompt_items.append(missing_values_rules[k])

        scaler_prompt = self.get_scaler_rules()
        if scaler_prompt is not None:
            prompt_items.append(scaler_prompt)

        # Encode categorical values:
        if self.flag_categorical_values and len(self.catalog.columns_categorical) > 0:
            categorical_column_prompt = (f'Encode categorical values by "on-hot-encoder" for the following '
                                         f'columns:\n\t# Columns: {",".join(self.catalog.columns_categorical)}')
            prompt_items.append(categorical_column_prompt)

        prompt_items.append(f"Dataset Attribute:\n# Number of samples (rows) in training dataset: {self.catalog.nrows}")
        prompt_items.append(f'Question: {self.question}')
        return f"\n\n{_user_delimiter}".join(prompt_items)

    def format_system_message(self):
        return "\n".join(self.rules)

    def format_schema_data(self):
        from util.Config import _user_delimiter
        content = []
        for r in range(0, len(self.df_content)):
            row_msg = [f'# {self.df_content.loc[r]["column_name"]} ({self.df_content.loc[r]["column_data_type"]})']
            if self.flag_distinct_value_count and self.df_content.loc[r]["is_categorical"] == False:
                row_msg.append(f'distinct-count [{self.df_content.loc[r]["distinct_count"]}]')

            if self.df_content.loc[r]["is_numerical"] and self.flag_statistical_number:
                row_msg.append(f'min-value [{self.df_content.loc[r]["min_value"]}]')
                row_msg.append(f'max-value [{self.df_content.loc[r]["max_value"]}]')
                row_msg.append(f'median-value [{self.df_content.loc[r]["median"]}]')
                row_msg.append(f'mean-value [{self.df_content.loc[r]["mean"]}]')

            if self.df_content.loc[r]["is_categorical"] and self.flag_categorical_values:
                row_msg.append(f'categorical-values [{self.df_content.loc[r]["categorical_values"]}]')

            if self.number_samples > 0 and self.flag_samples:
                row_msg.append(f'samples [{self.df_content.loc[r]["samples"]}]')

            content.append(", ".join(row_msg))
        content = "\n".join(content)
        prompt_items = [f"{self.ds_attribute_prefix_label}",
                        '"""',
                        content,
                        '"""']
        return "\n".join(prompt_items)

    def get_missing_values_rules(self):
        missing_values_rules = {"numerical_missing_values": None,
                                "bool_missing_values": None,
                                "categorical_missing_values": None,
                                "others_missing_values": None}

        if len(self.catalog.columns_numerical_missing_values) > 0:
            missing_values_prompt = (f"Do missing values imputation for the following numerical columns:\n\tColumns: "
                                     f"{','.join(self.catalog.columns_numerical_missing_values)}\n")
            missing_values_rules["numerical_missing_values"] = missing_values_prompt

        if len(self.catalog.columns_bool_missing_values) > 0:
            missing_values_prompt = (
                f"# Predict the missing values for the following boolean columns:\n\tColumns: "
                f"{','.join(self.catalog.columns_bool_missing_values)}")
            missing_values_rules["bool_missing_values"] = missing_values_prompt

        if len(self.catalog.columns_categorical_missing_values) > 0:
            missing_values_prompt = (f"# Predict the missing values for the following categorical columns:\n\tColumns: "
                                     f"{','.join(self.catalog.columns_categorical_missing_values)}\n")
            missing_values_rules["categorical_missing_values"] = missing_values_prompt

        if len(self.catalog.columns_others_missing_values) > 0:
            missing_values_prompt = (
                f"# Predict the missing values for the following string/object columns:\n\tColumns: "
                f"{','.join(self.catalog.columns_others_missing_values)}")
            missing_values_rules["others_missing_values"] = missing_values_prompt

        return missing_values_rules

    def get_scaler_rules(self):
        if self.number_samples > 0:
            return self.get_scaler_rules_few_shot()
        else:
            return self.get_scaler_rules_zero_shot()

    def get_scaler_rules_zero_shot(self):
        if len(self.catalog.columns_numerical_missing_values) > 0:
            scaler_prompt = (f"Select an appropriate scaler the following numerical columns "
                             f'(do it base on the min-max, mean, and median values are in the '
                             f'"Schema, and Data Profiling Info"):\n\t'
                             f"Columns: {','.join(self.catalog.columns_numerical_missing_values)}\n")
            return scaler_prompt
        else:
            return None

    def get_scaler_rules_few_shot(self):
        if len(self.catalog.columns_numerical_missing_values) > 0:
            scaler_prompt = (f"Select an appropriate scaler the following numerical columns "
                             f'(do it base on the min-max, mean, median, and samples values are in the '
                             f'"Schema, and Data Profiling Info"):\n\t'
                             f"Columns: {','.join(self.catalog.columns_numerical_missing_values)}\n")
            return scaler_prompt
        else:
            return None

    def get_drop_columns_rules(self):
        dropped_columns_names = self.catalog.drop_schema_info.keys()
        if len(dropped_columns_names) > 0:
            drop_column_prompt = "Drop the following column(s) from the train and test datasets:\n\tColumn(s): "
            names = []
            for k in dropped_columns_names:
                names.append(k)
            return f"{drop_column_prompt}{','.join(names)}\n"
        else:
            return None

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
        for k in self.catalog.schema_info.keys():
            if k in dropped_columns_names:
                continue

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
                categorical_values = (",".join([str(val) for val in cp.categorical_values])).replace("\"","'")
                tmp_cc = []
                for cv in cp.categorical_values:
                    tmp_cc.append(f"{cv}:{cp.categorical_values_ratio[str(cv)]}")

                categorical_values_ratio = (",".join(tmp_cc)).replace("\"","\'")

            if cp.samples is not None and len(cp.samples) > 0:
                samples_text = ",".join([str(val) for val in cp.samples[0:self.number_samples]])

            self.df_content.loc[len(self.df_content)] = [k, cp.short_data_type, cp.distinct_values_count, is_numerical,
                                                         cp.min_value, cp.max_value, cp.median, cp.mean, is_categorical,
                                                         categorical_values, categorical_values_ratio, samples_text]
