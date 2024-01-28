from util import StaticValues


class BasicPrompt(object):
    def __init__(self, *args, **kwargs):
        # used to avoid empty init function in 0-shot prompt
        pass

    def format_target(self, examples: dict):
        return {
            "rules": self.format_rules(),
            "question": self.format_question(examples=examples)
        }

    def format_question(self, examples: dict):
        prompt_items = [self.ds_attribute_prefix_label,
                        '"""',
                        self.content,
                        '"""\n',
                        f"Dataset Attribute:\nNumber of samples (rows) in training dataset: {self.nrows}\n",
                        f'Question: {self.question}']

        prompt = "\n".join(prompt_items)
        return prompt

    def format_rules(self):
        self.schema_keys = [_ for _ in self.schema.keys()]
        if self.task_type == "binary classification" or self.task_type == "multiclass classification":
            r54 = "classifier"
        else:
            r54 = "regressor"
        rules = [StaticValues.Rule_1.format(self.ds_attribute_prefix),
                 StaticValues.Rule_2.format(self.data_source_train_path, self.data_source_test_path),
                 StaticValues.Rule_3,
                 StaticValues.Rule_4.format(self.ds_attribute_prefix, self.ds_attribute_prefix_label),
                 StaticValues.Rule_5.format(f"{self.task_type}{self.suggested_model}", self.target_attribute,
                                            self.ds_attribute_prefix_label, r54, r54),
                 StaticValues.Rule_6,
                 StaticValues.CODE_FORMATTING_IMPORT,
                 StaticValues.CODE_FORMATTING_ADDING.format(self.target_attribute, self.schema_keys[0],
                                                            self.schema_keys[1]),
                 StaticValues.CODE_FORMATTING_DROPPING,
                 StaticValues.CODE_FORMATTING_TECHNIQUE,
                 self.evaluation_text,
                 "Don't report validation evaluation. We don't need it."
                 ]

        rule_msg = rules[0] + "\n\n"
        for i in range(1, len(rules)):
            rule_msg += f"Step {i}: {rules[i]}\n\n"

        return rule_msg


class SchemaPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        self.ds_attribute_prefix = "Schema"
        self.ds_attribute_prefix_label = "Schema:"
        self.question = ("Provide a complete pipeline code that can be executed in a multi-threaded environment "
                         "with various CPU configurations, such as PyTorch or other relevant frameworks.\n"
                         "Each codeblock ends with \"```end\" and starts with \"```python\".")

        self.content = "\n".join([f"{_} ({self.schema[_]})" for _ in self.schema.keys()])


class SchemaDistinctPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        self.ds_attribute_prefix = "Schema and Distinct Value Count"
        self.ds_attribute_prefix_label = "Schema and Distinct Value Count:"
        self.question = ("Provide a complete pipeline code that can be executed in a multi-threaded environment "
                         "with various CPU configurations, such as PyTorch or other relevant frameworks.\n"
                         "Each codeblock ends with \"```end\" and starts with \"```python\".")

        schema_info_list = []
        for k in self.schema.keys():
            cp = self.profile[k]
            str_val = f"{k} ({cp.short_data_type}): distinct-count [{cp.distinct_values_count}]"
            schema_info_list.append(str_val)

        self.content = "\n".join(schema_info_list)


class SchemaMissingValuePrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        self.ds_attribute_prefix = "Schema and Missing Value Frequency"
        self.ds_attribute_prefix_label = "Schema and Missing Value Frequency:"
        self.question = ("Provide a complete pipeline code that can be executed in a multi-threaded environment "
                         "with various CPU configurations, such as PyTorch or other relevant frameworks.\n"
                         "Each codeblock ends with \"```end\" and starts with \"```python\".")

        schema_info_list = []
        for k in self.schema.keys():
            cp = self.profile[k]
            str_val = f"{k} ({cp.short_data_type}): NaN-freq[{cp.missing_values_count / self.nrows:0.2f}%]"
            schema_info_list.append(str_val)

        self.content = "\n".join(schema_info_list)


class SchemaNumericStatisticPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        self.ds_attribute_prefix = "Schema and Numeric Statistical Values"
        self.ds_attribute_prefix_label = "Schema and Numeric Statistical Values:"
        self.question = ("Provide a complete pipeline code that can be executed in a multi-threaded environment "
                         "with various CPU configurations, such as PyTorch or other relevant frameworks.\n"
                         "Each codeblock ends with \"```end\" and starts with \"```python\".")

        schema_info_list = []
        for k in self.schema.keys():
            cp = self.profile[k]
            str_val = f"{k} ({cp.short_data_type})"
            if cp.data_type in {"int", "float"}:
                str_val += f": min-max values [{cp.min_value}, {cp.max_value}], mean [{cp.mean:0.2f}], median [{cp.median:0.2f}]"
            schema_info_list.append(str_val)

        self.content = "\n".join(schema_info_list)


class SchemaCategoricalValuePrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        self.ds_attribute_prefix = "Schema and Categorical Data"
        self.ds_attribute_prefix_label = "Schema and Categorical Data:"
        self.question = ("Provide a complete pipeline code that can be executed in a multi-threaded environment "
                         "with various CPU configurations, such as PyTorch or other relevant frameworks.\n"
                         "Each codeblock ends with \"```end\" and starts with \"```python\".")

        schema_info_list = []
        for k in self.schema.keys():
            cp = self.profile[k]
            r = cp.distinct_values_count / self.nrows
            categorical_column = ""
            if r <= 0.01:
                categorical_column = f": categorical column and distinct-count [{cp.distinct_values_count}]"
            str_val = f"{k} ({cp.short_data_type}){categorical_column}"
            schema_info_list.append(str_val)

        self.content = "\n".join(schema_info_list)


class SchemaDistinctMissingValuePrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        self.ds_attribute_prefix = "Schema, Distinct Value Count, and Missing Value Frequency"
        self.ds_attribute_prefix_label = "Schema, Distinct Value Count, and Missing Value Frequency:"
        self.question = ("Provide a complete pipeline code that can be executed in a multi-threaded environment "
                         "with various CPU configurations, such as PyTorch or other relevant frameworks.\n"
                         "Each codeblock ends with \"```end\" and starts with \"```python\".")

        schema_info_list = []
        for k in self.schema.keys():
            cp = self.profile[k]
            str_val = (f"{k} ({cp.short_data_type}): distinct-count [{cp.distinct_values_count}],  "
                       f"NaN-freq[{cp.missing_values_count / self.nrows:0.2f}%]")
            schema_info_list.append(str_val)

        self.content = "\n".join(schema_info_list)


class SchemaDistinctNumericStatisticPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        self.ds_attribute_prefix = "Schema, Distinct Value Count, and Numeric Statistical Values"
        self.ds_attribute_prefix_label = "Schema, Distinct Value Count, and Numeric Statistical Values:"
        self.question = ("Provide a complete pipeline code that can be executed in a multi-threaded environment "
                         "with various CPU configurations, such as PyTorch or other relevant frameworks.\n"
                         "Each codeblock ends with \"```end\" and starts with \"```python\".")

        schema_info_list = []
        for k in self.schema.keys():
            cp = self.profile[k]
            str_val = f"{k} ({cp.short_data_type}): distinct-count [{cp.distinct_values_count}]"

            if cp.data_type in {"int", "float"}:
                str_val += f": min-max values [{cp.min_value}, {cp.max_value}], mean [{cp.mean:0.2f}], median [{cp.median:0.2f}]"
            schema_info_list.append(str_val)

        self.content = "\n".join(schema_info_list)


class SchemaMissingValueNumericStatisticPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        self.ds_attribute_prefix = "Schema, Missing Value Frequency, and Numeric Statistical Values"
        self.ds_attribute_prefix_label = "Schema, Missing Value Frequency, and Numeric Statistical Values:"
        self.question = ("Provide a complete pipeline code that can be executed in a multi-threaded environment "
                         "with various CPU configurations, such as PyTorch or other relevant frameworks.\n"
                         "Each codeblock ends with \"```end\" and starts with \"```python\".")

        schema_info_list = []
        for k in self.schema.keys():
            cp = self.profile[k]
            str_val = f"{k} ({cp.short_data_type}): NaN-freq[{cp.missing_values_count / self.nrows:0.2f}%]"

            if cp.data_type in {"int", "float"}:
                str_val += f", min-max values [{cp.min_value}, {cp.max_value}], mean [{cp.mean:0.2f}], median [{cp.median:0.2f}]"
            schema_info_list.append(str_val)

        self.content = "\n".join(schema_info_list)


class SchemaMissingCategoricalValuePrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        self.ds_attribute_prefix = "Schema, Missing Value Frequency, and Categorical Data"
        self.ds_attribute_prefix_label = "Schema, Missing Value Frequency, and Numeric Statistical Data:"
        self.question = ("Provide a complete pipeline code that can be executed in a multi-threaded environment "
                         "with various CPU configurations, such as PyTorch or other relevant frameworks.\n"
                         "Each codeblock ends with \"```end\" and starts with \"```python\".")

        schema_info_list = []
        for k in self.schema.keys():
            cp = self.profile[k]
            str_val = f"{k} ({cp.short_data_type}): NaN-freq[{cp.missing_values_count / self.nrows:0.2f}%]"
            r = cp.distinct_values_count / self.nrows
            if r <= 0.01:
                str_val += f", categorical column and distinct-count [{cp.distinct_values_count}]"
            schema_info_list.append(str_val)

        self.content = "\n".join(schema_info_list)


class SchemaNumericStatisticCategoricalValuePrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        self.ds_attribute_prefix = "Schema, Numeric Statistical Values, and Categorical Data"
        self.ds_attribute_prefix_label = "Schema, Numeric Statistical Values, and Categorical Data:"
        self.question = ("Provide a complete pipeline code that can be executed in a multi-threaded environment "
                         "with various CPU configurations, such as PyTorch or other relevant frameworks.\n"
                         "Each codeblock ends with \"```end\" and starts with \"```python\".")

        schema_info_list = []
        for k in self.schema.keys():
            cp = self.profile[k]
            r = cp.distinct_values_count / self.nrows
            str_val = f"{k} ({cp.short_data_type})"
            colon_flag = False
            if r <= 0.01:
                str_val += f": categorical column and distinct-count [{cp.distinct_values_count}]"
                colon_flag = True
            if cp.data_type in {"int", "float"}:
                if not colon_flag:
                    str_val += ": "
                else:
                    str_val +=", "
                str_val += f"min-max values [{cp.min_value}, {cp.max_value}], mean [{cp.mean:0.2f}], median [{cp.median:0.2f}]"
            schema_info_list.append(str_val)

        self.content = "\n".join(schema_info_list)


class AllPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        self.ds_attribute_prefix = "Schema, and Data Profiling Info"
        self.ds_attribute_prefix_label = "Schema, and Data Profiling Info:"
        self.question = ("Provide a complete pipeline code that can be executed in a multi-threaded environment "
                         "with various CPU configurations, such as PyTorch or other relevant frameworks.\n"
                         "Each codeblock ends with \"```end\" and starts with \"```python\".")

        schema_info_list = []
        for k in self.schema.keys():
            cp = self.profile[k]
            r = cp.distinct_values_count / self.nrows
            str_val = (f"{k} ({cp.short_data_type}): distinct-count [{cp.distinct_values_count}], "
                       f"NaN-freq[{cp.missing_values_count / self.nrows:0.2f}%")

            if r <= 0.01:
                str_val += f", categorical column"

            if cp.data_type in {"int", "float"}:
                str_val += f", min-max values [{cp.min_value}, {cp.max_value}], mean [{cp.mean:0.2f}], median [{cp.median:0.2f}]"
            schema_info_list.append(str_val)

        self.content = "\n".join(schema_info_list)