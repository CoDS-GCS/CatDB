import re


class BasicPrompt(object):
    def __init__(self, *args, **kwargs):
        # used to avoid empty init function in 0-shot prompt
        pass

    def format_target(self):
        return self.format_question()

    def format_question(self, examples: dict):
        raise NotImplementedError()

    def get_extra_info(self):
        return {"evaluation_text": self.evaluation_text,
                "sample_attribute_names": [self.schema_keys[0], self.schema_keys[1]],
                "task_type": self.task_type,
                "target_attribute": self.target_attribute,
                "data_source_train_path": self.data_source_train_path,
                "data_source_test_path": self.data_source_test_path
                }


class SchemaPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        self.schema_str = "\n".join([f"{_} (data type:{self.schema[_]})" for _ in self.schema.keys()])
        self.schema_keys = [_ for _ in self.schema.keys()]

        self.template_question = "Generate as many features as useful for downstream classifier, \
        but as few as necessary to reach good performance. and can drop unused columns (Feature selection).\
        \n Each codeblock ends with \"```end\" and starts with \"```python\" \n Return a full pipeline code."

    def format_question(self):
        prefix_key = "Schema:"
        schema_rule = f"the user will provide the schema of the dataset with columns appropriately named as attributes, enclosed in triple quotes, and preceded by the prefix {prefix_key}."

        return {"question": re.sub(' +', ' ', self.template_question),
                "prompt": re.sub(' +', ' ', self.schema_str),
                "rule": schema_rule,
                "prefix_key": prefix_key}
