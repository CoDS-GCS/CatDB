from util import StaticValues
import pandas as pd


class BasicPrompt(object):
    def __init__(self, *args, **kwargs):
        self.ds_attribute_prefix = None
        self.ds_attribute_prefix_label = None
        self.rules = []
        self.schema_keys = None
        self.question = None
        self.extra_info = None
        self.df_content = None
        self.set_schema_content()

    def format_prompt(self):
        return {
            "system_message": self.format_system_message(),
            "user_message": self.format_user_message()
        }

    def format_user_message(self):
        prompt_items = []
        if self.dataset_description is not None:
            prompt_items.append(StaticValues.dataset_description.format(self.dataset_description))

        prompt_items.append(self.format_schema_data())

        missing_values_rules = self.get_missing_values_rules()
        for k in missing_values_rules.keys():
            if missing_values_rules[k] is not None:
                prompt_items.append(missing_values_rules[k])

        scaler_prompt = self.get_scaler_rules()
        if scaler_prompt is not None:
            prompt_items.append(scaler_prompt)

        prompt_items.append(f"Dataset Attribute:\nNumber of samples (rows) in training dataset: {self.nrows}")
        prompt_items.append(f'Question: {self.question}')
        return prompt_items

    def format_system_message(self):
        self.schema_keys = [_ for _ in self.schema.keys()]
        if self.task_type == "binary classification" or self.task_type == "multiclass classification":
            algorithm = "classifier"
        else:
            algorithm = "regressor"

        rule_msg = "\n".join(self.rules)
        return rule_msg

    def format_schema_data(self):
        content = []
        for r in len(self.df_content):
            row_msg = []
            row_msg.append(f' #{self.df_content[r]["column_name"]} ({self.df_content[r]["column_data_type"]})')
            row_msg.append(f'distinct-count [{self.df_content[r]["distinct_count"]}]')
            if self.df_content[r]["is_numerical"]:
                row_msg.append(f'min-value [{self.df_content[r]["min_value"]}]')
                row_msg.append(f'max-value [{self.df_content[r]["max_value"]}]')
                row_msg.append(f'median-value [{self.df_content[r]["median"]}]')
                row_msg.append(f'mean-value [{self.df_content[r]["mean"]}]')

            if self.df_content[r]["is_categorical"]:
                row_msg.append(f'categorical-values-count [{self.df_content[r]["categorical_values_ratio"]}]')

            if self.number_samples > 0:
                row_msg.append(f'samples [{self.df_content[r]["samples"]}]')

            content.append(",".join(row_msg))
        content = "\n".join(content)
        prompt_items = [self.ds_attribute_prefix_label,
                        '"""',
                        content,
                        '"""\n']
        return prompt_items

    def get_missing_values_rules(self):
        missing_values_rules = {"numerical_missing_values": None,
                                "bool_missing_values": None,
                                "categorical_missing_values": None,
                                "others_missing_values": None}

        if len(self.profile.columns_numerical_missing_values) > 0:
            missing_values_prompt = (f"Do missing values imputation for the following numerical columns:\n\tColumns: "
                                     f"{','.join(self.profile.columns_numerical_missing_values)}\n")
            missing_values_rules["numerical_missing_values"] = missing_values_prompt

        if len(self.profile.columns_bool_missing_values) > 0:
            missing_values_prompt = (
                f"# Predict the missing values for the following boolean columns:\n\tColumns: "
                f"{','.join(self.profile.columns_bool_missing_values)}")
            missing_values_rules["bool_missing_values"] = missing_values_prompt

        if len(self.profile.columns_categorical_missing_values) > 0:
            missing_values_prompt = (f"# Predict the missing values for the following categorical columns:\n\tColumns: "
                                     f"{','.join(self.profile.columns_categorical_missing_values)}\n")
            missing_values_rules["categorical_missing_values"] = missing_values_prompt

        if len(self.profile.columns_others_missing_values) > 0:
            missing_values_prompt = (
                f"# Predict the missing values for the following string/object columns:\n\tColumns: "
                f"{','.join(self.profile.columns_others_missing_values)}")
            missing_values_rules["others_missing_values"] = missing_values_prompt

        return missing_values_rules

    def get_scaler_rules(self):
        if self.number_samples > 0:
            return self.get_scaler_rules_few_shot()
        else:
            return self.get_scaler_rules_zero_shot()

    def get_scaler_rules_zero_shot(self):
        if len(self.profile.columns_numerical_missing_values) > 0:
            scaler_prompt = (f"Select an appropriate scaler the following numerical columns "
                             f'(do it base on the min-max, mean, and median values are in the '
                             f'"Schema, and Data Profiling Info"):\n\t'
                             f"Columns: {','.join(self.profile.columns_numerical_missing_values)}\n")
            return scaler_prompt
        else:
            return None

    def get_scaler_rules_few_shot(self):
        if len(self.profile.columns_numerical_missing_values) > 0:
            scaler_prompt = (f"Select an appropriate scaler the following numerical columns "
                             f'(do it base on the min-max, mean, median, and samples values are in the '
                             f'"Schema, and Data Profiling Info"):\n\t'
                             f"Columns: {','.join(self.profile.columns_numerical_missing_values)}\n")
            return scaler_prompt
        else:
            return None

    def get_drop_columns_rules(self):
        dropped_columns_names = self.dropped_columns.keys()
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
        dropped_columns_names = self.dropped_columns.keys()
        for k in self.schema.keys():
            if k in dropped_columns_names:
                continue

            cp = self.profile[k]

            is_numerical = False
            is_categorical = False
            categorical_values = None
            categorical_values_ratio = None
            samples_text = None

            if k in self.profile.column_numerical:
                is_numerical = True

            if k in self.profile.column_categorical:
                is_categorical = True
                categorical_values = ",".join(cp.category_values)
                tmp_cc = []
                for cv in cp.category_values:
                    tmp_cc.append(f"{cv}:{cp.category_values_ratio[cv]}")

                categorical_values_ratio = ",".join(tmp_cc)

            if cp.samples is not None and len(cp.samples) > 0:
                samples_text = ",".join(cp.samples[0:self.number_samples])

            self.df_content.loc[len(self.df_content)] = [cp.short_data_type, cp.distinct_values_count, is_numerical,
                                                         cp.min_value, cp.max_value, cp.median, cp.mean, is_categorical,
                                                         categorical_values, categorical_values_ratio, samples_text]
