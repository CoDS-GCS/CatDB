from util import StaticValues
from util import Config


class BasicPrompt(object):
    def __init__(self, *args, **kwargs):
        self.schema_keys = None
        self.question = ("Provide a complete pipeline code that can be executed in a multi-threaded environment "
                         "with various CPU configurations, such as PyTorch or other relevant frameworks.\n"
                         "Each codeblock ends with \"```end\" and starts with \"```python\".")
        self.extra_info = ""

    def format_target(self, examples: dict):
        return {
            "rules": self.format_rules(),
            "question": self.format_question(examples=examples)
        }

    def format_question(self, examples: dict):
        prompt_items = []
        if self.dataset_description is not None:
            prompt_items.append(StaticValues.dataset_description.format(self.dataset_description))

        prompt_items.extend([self.ds_attribute_prefix_label,
                        '"""',
                        self.content,
                        '"""\n',
                        f"Dataset Attribute:\nNumber of samples (rows) in training dataset: {self.nrows}\n",
                        self.extra_info,
                        f'Question: {self.question}'])

        prompt = "\n".join(prompt_items)
        return prompt

    def format_rules(self):
        self.schema_keys = [_ for _ in self.schema.keys()]
        if self.task_type == "binary classification" or self.task_type == "multiclass classification":
            algorithm = "classifier"
        else:
            algorithm = "regressor"

        randomforest_param = int (self.nrows * 0.1)
        if randomforest_param < 3000:
            randomforest_param = self.nrows
        rules = [StaticValues.Rule_task.format(self.ds_attribute_prefix),
                 StaticValues.Rule_input,
                 StaticValues.Rule_output,
                 f"\t 1. {StaticValues.Rule_1}",
                 f"\t 2. {StaticValues.Rule_2.format(self.data_source_train_path, self.data_source_test_path)}",
                 f"\t 3. {StaticValues.Rule_3}",
                 f"\t 4. {StaticValues.Rule_4.format(self.ds_attribute_prefix, self.ds_attribute_prefix_label)}",
                 f"\t 5. {StaticValues.Rule_5}",
                 f"\t 6. {StaticValues.Rule_12}",
                 f"\t 7. {StaticValues.Rule_6}",
                 f"\t 8. {StaticValues.Rule_7.format(self.target_attribute)}",
                 f"\t 9. {StaticValues.Rule_8.format(algorithm, self.target_attribute)}",
                 f"\t 10. {StaticValues.Rule_9}",
                 f"\t 11. {StaticValues.CODE_FORMATTING_IMPORT}",
                 f"\t 12. {StaticValues.CODE_FORMATTING_ADDING.format(self.target_attribute, self.schema_keys[0], self.schema_keys[1])}",
                 f"\t 13. {StaticValues.CODE_FORMATTING_DROPPING}",
                 f"\t 14. {StaticValues.CODE_FORMATTING_TECHNIQUE.format(algorithm)}",
                 f"\t 15. {self.evaluation_text}",
                 f"\t 16. {StaticValues.Rule_10}",
                 f"\t 17. {StaticValues.Rule_11}",
                 f"\t 18. {StaticValues.Rule_12.format(randomforest_param)}"
                 ]

        rule_msg = "\n".join(rules)
        return rule_msg


class SchemaPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        BasicPrompt.__init__(self, *args, **kwargs)
        self.ds_attribute_prefix = "Schema"
        self.ds_attribute_prefix_label = "Schema:"
        self.content = "\n".join([f"{_} ({self.schema[_]})" for _ in self.schema.keys()])


class SchemaDistinctValuePrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        BasicPrompt.__init__(self, *args, **kwargs)
        self.ds_attribute_prefix = "Schema and Distinct Value Count"
        self.ds_attribute_prefix_label = "Schema and Distinct Value Count:"
        schema_info_list = []
        for k in self.schema.keys():
            cp = self.profile[k]
            str_val = f"{k} ({cp.short_data_type}): distinct-count [{cp.distinct_values_count}]"
            schema_info_list.append(str_val)

        self.content = "\n".join(schema_info_list)


class SchemaMissingValueFrequencyPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        BasicPrompt.__init__(self, *args, **kwargs)
        self.ds_attribute_prefix = "Schema and Missing Value Frequency"
        self.ds_attribute_prefix_label = "Schema and Missing Value Frequency:"
        schema_info_list = []
        for k in self.schema.keys():
            cp = self.profile[k]
            str_val = f"{k} ({cp.short_data_type}): NaN-freq[{(cp.missing_values_count / self.nrows) * 100:0.2f}%]"
            schema_info_list.append(str_val)

        self.content = "\n".join(schema_info_list)


class SchemaStatisticNumericPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        BasicPrompt.__init__(self, *args, **kwargs)
        self.ds_attribute_prefix = "Schema and Numeric Statistical Values"
        self.ds_attribute_prefix_label = "Schema and Numeric Statistical Values:"

        schema_info_list = []
        for k in self.schema.keys():
            cp = self.profile[k]
            str_val = f"{k} ({cp.short_data_type})"
            if cp.data_type in {"int", "float"} and cp.distinct_values_count > 0:
                str_val += f": min-max values [{cp.min_value}, {cp.max_value}], mean [{cp.mean:0.2f}], median [{cp.median:0.2f}]"
            schema_info_list.append(str_val)

        self.content = "\n".join(schema_info_list)


class SchemaCategoricalValuesPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        BasicPrompt.__init__(self, *args, **kwargs)
        self.ds_attribute_prefix = "Schema and Categorical Data"
        self.ds_attribute_prefix_label = "Schema and Categorical Data:"

        schema_info_list = []
        for k in self.schema.keys():
            cp = self.profile[k]
            r = cp.distinct_values_count / self.nrows
            categorical_column = ""
            if r <= Config.CATEGORICAL_RATIO:
                categorical_column = f": categorical column and distinct-count [{cp.distinct_values_count}]"
            str_val = f"{k} ({cp.short_data_type}){categorical_column}"
            schema_info_list.append(str_val)

        self.content = "\n".join(schema_info_list)


class SchemaDistinctValueCountMissingValueFrequencyPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        BasicPrompt.__init__(self, *args, **kwargs)
        self.ds_attribute_prefix = "Schema, Distinct Value Count, and Missing Value Frequency"
        self.ds_attribute_prefix_label = "Schema, Distinct Value Count, and Missing Value Frequency:"

        schema_info_list = []
        for k in self.schema.keys():
            cp = self.profile[k]
            str_val = (f"{k} ({cp.short_data_type}): distinct-count [{cp.distinct_values_count}],  "
                       f"NaN-freq[{(cp.missing_values_count / self.nrows) * 100:0.2f}%]")
            schema_info_list.append(str_val)

        self.content = "\n".join(schema_info_list)


class SchemaDistinctValueCountStatisticNumericPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        BasicPrompt.__init__(self, *args, **kwargs)
        self.ds_attribute_prefix = "Schema, Distinct Value Count, and Numeric Statistical Values"
        self.ds_attribute_prefix_label = "Schema, Distinct Value Count, and Numeric Statistical Values:"

        schema_info_list = []
        for k in self.schema.keys():
            cp = self.profile[k]
            str_val = f"{k} ({cp.short_data_type}): distinct-count [{cp.distinct_values_count}]"

            if cp.data_type in {"int", "float"} and cp.distinct_values_count > 0:
                str_val += f": min-max values [{cp.min_value}, {cp.max_value}], mean [{cp.mean:0.2f}], median [{cp.median:0.2f}]"
            schema_info_list.append(str_val)

        self.content = "\n".join(schema_info_list)


class SchemaMissingValueFrequencyStatisticNumericPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        BasicPrompt.__init__(self, *args, **kwargs)
        self.ds_attribute_prefix = "Schema, Missing Value Frequency, and Numeric Statistical Values"
        self.ds_attribute_prefix_label = "Schema, Missing Value Frequency, and Numeric Statistical Values"
        schema_info_list = []
        for k in self.schema.keys():
            cp = self.profile[k]
            str_val = f"{k} ({cp.short_data_type}): NaN-freq[{(cp.missing_values_count / self.nrows) * 100:0.2f}%]"

            if cp.data_type in {"int", "float"} and cp.distinct_values_count > 0:
                str_val += f", min-max values [{cp.min_value}, {cp.max_value}], mean [{cp.mean:0.2f}], median [{cp.median:0.2f}]"
            schema_info_list.append(str_val)

        self.content = "\n".join(schema_info_list)


class SchemaMissingValueFrequencyCategoricalValuesPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        BasicPrompt.__init__(self, *args, **kwargs)
        self.ds_attribute_prefix = "Schema, Missing Value Frequency, and Categorical Data"
        self.ds_attribute_prefix_label = "Schema, Missing Value Frequency, and Categorical Data"

        schema_info_list = []
        for k in self.schema.keys():
            cp = self.profile[k]
            str_val = f"{k} ({cp.short_data_type}): NaN-freq[{(cp.missing_values_count / self.nrows) * 100:0.2f}%]"
            r = cp.distinct_values_count / self.nrows
            if r <= Config.CATEGORICAL_RATIO:
                str_val += f", categorical column and distinct-count [{cp.distinct_values_count}]"
            schema_info_list.append(str_val)

        self.content = "\n".join(schema_info_list)


class SchemaStatisticNumericCategoricalValuesPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        BasicPrompt.__init__(self, *args, **kwargs)
        self.ds_attribute_prefix = "Schema, Numeric Statistical Values, and Categorical Data"
        self.ds_attribute_prefix_label = "Schema, Numeric Statistical Values, and Categorical Data:"

        schema_info_list = []
        for k in self.schema.keys():
            cp = self.profile[k]
            r = cp.distinct_values_count / self.nrows
            str_val = f"{k} ({cp.short_data_type})"
            colon_flag = False
            if r <= Config.CATEGORICAL_RATIO:
                str_val += f": categorical column and distinct-count [{cp.distinct_values_count}]"
                colon_flag = True
            if cp.data_type in {"int", "float"} and cp.distinct_values_count > 0:
                if not colon_flag:
                    str_val += ": "
                else:
                    str_val += ", "
                str_val += f"min-max values [{cp.min_value}, {cp.max_value}], mean [{cp.mean:0.2f}], median [{cp.median:0.2f}]"
            schema_info_list.append(str_val)

        self.content = "\n".join(schema_info_list)


class AllPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        BasicPrompt.__init__(self, *args, **kwargs)
        self.ds_attribute_prefix = "Schema, and Data Profiling Info"
        self.ds_attribute_prefix_label = "Schema, and Data Profiling Info:"

        schema_info_list = []
        for k in self.schema.keys():
            cp = self.profile[k]
            r = cp.distinct_values_count / self.nrows
            str_val = (f"{k} ({cp.short_data_type}): distinct-count [{cp.distinct_values_count}], "
                       f"NaN-freq[{cp.missing_values_count / self.nrows:0.2f}%]")

            if r <= Config.CATEGORICAL_RATIO:
                str_val += f", categorical column"

            if cp.data_type in {"int", "float"} and cp.distinct_values_count > 0:
                str_val += f", min-max values [{cp.min_value}, {cp.max_value}], mean [{cp.mean:0.2f}], median [{cp.median:0.2f}]"
            schema_info_list.append(str_val)

        self.content = "\n".join(schema_info_list)


class CatDBPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        BasicPrompt.__init__(self, *args, **kwargs)
        self.ds_attribute_prefix = "Schema, and Data Profiling Info"
        self.ds_attribute_prefix_label = "Schema, and Data Profiling Info:"
        extra_info_items = []

        schema_info_list = []
        dropped_columns_names = self.dropped_columns.keys()
        missing_values_columns = []
        none_numerical_missing_values_columns = []
        categorical_missing_values_column = []
        categorical_columns = []
        numerical_columns = []

        # Drop unnecessary columns:
        if len(dropped_columns_names) > 0:
            drop_column_prompt = "# Drop the following column(s) from the train and test datasets:\n\tColumn(s): "
            names = []
            for k in dropped_columns_names:
                names.append(k)
            drop_column_prompt = f"{drop_column_prompt}{','.join(names)}\n"
            extra_info_items.append(drop_column_prompt)

        # Find missing value, categorical, and numerical columns and do missing value imputation:
        for k in self.schema.keys():
            if k in dropped_columns_names or k == self.target_attribute:
                continue

            cp = self.profile[k]
            r = cp.distinct_values_count / self.nrows

            if 0 < cp.missing_values_count < self.nrows:
                if r > Config.CATEGORICAL_RATIO and cp.data_type != "named_entity":
                    if cp.data_type in {"int", "float"}:
                        missing_values_columns.append(k)
                    else:
                        none_numerical_missing_values_columns.append(k)
                else:
                    categorical_missing_values_column.append(k)

            if r <= Config.CATEGORICAL_RATIO or cp.data_type == "named_entity":
                categorical_columns.append(k)

            if cp.data_type in {"int", "float"} and cp.distinct_values_count > 0:
                numerical_columns.append(k)

        # Missing value imputation:
        if len(missing_values_columns) > 0:
            missing_values_prompt = (f"# Do missing values imputation for the following numerical columns:\n\tColumns: "
                                     f"{','.join(missing_values_columns)}\n")
            extra_info_items.append(missing_values_prompt)

        if len(none_numerical_missing_values_columns) > 0:
            missing_values_prompt = (
                f"# Predict the missing values for the following none-numerical columns:\n\tColumns: "
                f"{','.join(none_numerical_missing_values_columns)}\n")
            extra_info_items.append(missing_values_prompt)

        if len(categorical_missing_values_column) > 0:
            missing_values_prompt = (f"# Predict the missing values for the following categorical columns:\n\tColumns: "
                                     f"{','.join(categorical_missing_values_column)}\n")
            extra_info_items.append(missing_values_prompt)

        # Add data scaler
        if len(numerical_columns) > 0:
            numerical_column_prompt = (f"# Select an appropriate scaler the following numerical columns "
                                       f'(do it base on the min-max, mean, and median values are in the '
                                       f'"Schema, and Data Profiling Info"):\n\t'
                                       f"Columns: {','.join(numerical_columns)}\n")
            extra_info_items.append(numerical_column_prompt)

        # Encode categorical values:
        if len(categorical_columns) > 0:
            categorical_column_prompt = (f'# Encode categorical values by "on-hot-encoder" for the following columns:'
                                         f"\n\tColumns: {','.join(categorical_columns)}\n")
            extra_info_items.append(categorical_column_prompt)

        extra_info_items.append('# Encode all "object" columns by dummyEncode.\n\n')

        for k in self.schema.keys():
            if k in dropped_columns_names:
                continue

            cp = self.profile[k]
            str_val = f"{k} ({cp.short_data_type}): distinct-count [{cp.distinct_values_count}]"

            if k in numerical_columns:
                str_val += f", min-max values [{cp.min_value}, {cp.max_value}], mean [{cp.mean:0.2f}], median [{cp.median:0.2f}]"

            schema_info_list.append(str_val)

        self.content = "\n".join(schema_info_list)

        if len(extra_info_items) > 0:
            self.extra_info = "".join(extra_info_items)
        else:
            self.extra_info = ""
