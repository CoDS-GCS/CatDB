from .BasicPrompt import BasicPrompt
import pandas as pd


class MissingValuePrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ds_attribute_prefix = "Schema, and Data Profiling Info"
        self.ds_attribute_prefix_label = "Schema, and Data Profiling Info:"
        self.question = None
        self.config = None

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

    def format_system_message(self):
        from util.Config import _system_delimiter, _catdb_rules
        self.schema_keys = [_ for _ in self.catalog.schema_info.keys()]
        if self.config == "CatDB":
            rules = [f"{_system_delimiter} {_catdb_rules['task'].format(self.ds_attribute_prefix)}",
                     f"{_system_delimiter} {_catdb_rules['input']}",
                     f"{_system_delimiter} {_catdb_rules['output']}",
                     f"# 1: {_catdb_rules['Rule_1'].format(self.ds_attribute_prefix, self.ds_attribute_prefix_label)}",
                     f"# 2: {_catdb_rules['Rule_2']}",
                     f"# 3: {_catdb_rules['Rule_3']}",
                     f"# 4: {_catdb_rules['Rule_4']}",
                     f"# 5: {_catdb_rules['Rule_5']}"]
        else:
            rules = []
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

class CatDBMissingValuePrompt(MissingValuePrompt):
    def __init__(self, *args, **kwargs):
        MissingValuePrompt.__init__(self,
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