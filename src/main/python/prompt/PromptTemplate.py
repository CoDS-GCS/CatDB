from src.main.python.catalog.Catalog import CatalogInfo
from src.main.python.util import StaticValues
import re


class BasicPrompt(object):
    def __init__(self, *args, **kwargs):
        # used to avoid empty init function in 0-shot prompt
        pass

    def format_target(self, example: dict):
        return self.format_question(example) + "\nSELECT "

    def format_question(self, examples: dict):
        raise NotImplementedError()

    def get_extra_info(self, examples: dict):
        return None


class TextPrompt(BasicPrompt):
    def __init__(self, *args, **kwargs):
        self.template_info = "The dataframe `df` has been successfully loaded into memory, with columns appropriately named as attributes.\n" \
                             "Columns in `df` (true feature dtypes listed here):\n" \
                             "{}"
        self.template_question = "\n" + StaticValues.PROMPT_ADDITIONAL_TEXT + "\n Answer the following: {}"

        self.how_many = (
            "up to 10 useful columns. Generate as many features as useful for downstream classifier, but as few as necessary to reach good performance."
            if self.iterative == 1
            else "exactly one useful column"
        )

    def format_question(self, examples: dict):
        schema = "\n".join([f"{_} ({self.schema[_]})" for _ in self.schema.keys()])
        schema_keys = [_ for _ in self.schema.keys()]
        prompt = f"""
            The dataframe `df` has been successfully loaded into memory, with columns appropriately named as attributes: \n{schema}
            
            This code was written by an expert datascientist working to improve predictions. It is a snippet of code that adds new columns to the dataset.
       
        This code generates additional columns that are useful for a downstream classification algorithm (such as XGBoost) predicting \"{self.target_attribute}\".
        Additional columns add new semantic information, that is they use real world knowledge on the dataset. They can e.g. be feature combinations, transformations, aggregations where the new column is a function of the existing columns.
        The scale of columns and offset does not matter. Make sure all used columns exist. Follow the above description of columns closely and consider the datatypes and meanings of classes.
        This code also drops columns, if these may be redundant and hurt the predictive performance of the downstream classifier (Feature selection). Dropping columns may help as the chance of overfitting is lower, especially if the dataset is small.
        The classifier will be trained on the dataset with the generated columns and evaluated on a holdout set. The evaluation metric is accuracy. The best performing code will be selected.
        Added columns can be used in other codeblocks, dropped columns are not available anymore.

        Code formatting for each added column:
        ```python
        # (Feature name and description)
        # Usefulness: (Description why this adds useful real world knowledge to classify \"{self.target_attribute}\" according to dataset description and attributes.)
        (Some pandas code using '{schema_keys[0]}', '{schema_keys[1]}', ... to add a new column for each row in df)
        ```end

        Code formatting for dropping columns:
        ```python
        # Explanation why the column XX is dropped
        df.drop(columns=['XX'], inplace=True)
        ```end

        Each codeblock generates {self.how_many} and can drop unused columns (Feature selection).
        Each codeblock ends with ```end and starts with "```python"
        Codeblock:
        """

        return re.sub(' +', ' ', prompt)

# class SQLPrompt(BasicPrompt):
#     template_info =   "/* Given the following database schema: */\n" \
#                       "{}"
#     template_question =  "/* Answer the following: {} */"
#
#     def format_question(self, example: dict):
#         sqls = get_sql_for_database(example["path_db"])
#
#         prompt_info = self.template_info.format("\n\n".join(sqls))
#         prompt_extra_info = self.get_extra_info(example["db_id"])
#         prompt_question = self.template_question.format(example["question"])
#
#         if prompt_extra_info is None or prompt_extra_info == "":
#             prompt_components = [prompt_info, prompt_question]
#         else:
#             prompt_components = [prompt_info, prompt_extra_info, prompt_question]
#
#         prompt = "\n\n".join(prompt_components)
#         return prompt
#
#
# class TextPrompt(BasicPrompt):
#     template_info = "Given the following database schema:\n" \
#                   "{}"
#     template_question = "Answer the following: {}"
#
#     def format_question(self, example: dict):
#         schemas = "\n".join([f"{_.name}: {', '.join(_.schema)}" for _ in example["tables"]])
#
#         prompt_info = self.template_info.format(schemas)
#         prompt_extra_info = self.get_extra_info(example["db_id"])
#         prompt_question = self.template_question.format(example["question"])
#
#         if prompt_extra_info is None or prompt_extra_info == "":
#             prompt_components = [prompt_info,prompt_question]
#         else:
#             prompt_components = [prompt_info, prompt_extra_info, prompt_question]
#
#         prompt = "\n".join(prompt_components)
#         return prompt
#
#
# class NumberSignPrompt(BasicPrompt):
#     template_info = "### Complete sqlite SQL query only and with no explanation\n" \
#                     "### SQLite SQL tables, with their properties:\n" \
#                     "#\n" \
#                     "{}\n" \
#                     "#"
#     template_question = "### {}"
#
#     def format_question(self, example: dict):
#         schemas = "\n".join([f"# {_.name}({', '.join(_.schema)})" for _ in example["tables"]])
#
#         prompt_info = self.template_info.format(schemas)
#         prompt_extra_info = self.get_extra_info(example["db_id"])
#         prompt_question = self.template_question.format(example["question"])
#
#         if prompt_extra_info is None or prompt_extra_info == "":
#             prompt_components = [prompt_info,prompt_question]
#         else:
#             prompt_components = [prompt_info, prompt_extra_info, prompt_question]
#
#         prompt = "\n".join(prompt_components)
#         return prompt
#
#
# class BaselinePrompt(BasicPrompt):
#     template_info = "{}\nForeign_keys={}\n"
#     template_question = "Q: \"{}\""
#
#     def format_question(self, example: dict):
#         # schemas
#         schemas = "\n".join([f"Table {_.name}, columns = {_.schema}" for _ in example["tables"]]).replace("'", "")
#         # foreign_keys
#         foreign_keys = list()
#         for table in example["tables"]:
#             for pair_str in table["table_info"]["foreign_key"]:
#                 a, b = [_.strip() for _ in pair_str[1:-1].split(",")]
#                 foreign_keys.append(f"{a}={b}")
#
#         # format prompt
#         prompt_info = self.template_info.format(schemas, str(foreign_keys).replace("'", ""))
#         prompt_extra_info = self.get_extra_info(example["db_id"])
#         prompt_question = self.template_question.format(example["question"])
#
#         if prompt_extra_info is None or prompt_extra_info == "":
#             prompt_components = [prompt_info,prompt_question]
#         else:
#             prompt_components = [prompt_info, prompt_extra_info, prompt_question]
#
#         prompt = "".join(prompt_components)
#         return prompt
#
#     def format_target(self, example: dict):
#         return self.format_question(example) + "\nA: SELECT "
#
#
# class InstructionPrompt(BasicPrompt):
#     template_info = (
#         "Below is an instruction that describes a task, paired with an input that provides further context. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\nWrite a sql to answer the question \"{}\"\n\n### Input:\n{}\n"
#     )
#     template_question = "### Response:"
#
#     def format_question(self, example: dict):
#         schemas = "\n".join([f"{_.name}({', '.join(_.schema)})" for _ in example["tables"]])
#
#         prompt_info = self.template_info.format(example["question"], schemas)
#         prompt_extra_info = self.get_extra_info(example["db_id"])
#         prompt_question = self.template_question
#
#         if prompt_extra_info is None or prompt_extra_info == "":
#             prompt_components = [prompt_info, prompt_question]
#         else:
#             # TODO: extra_info should be after info
#             prompt_components = [prompt_info, prompt_extra_info, prompt_question]
#
#         prompt = "\n".join(prompt_components)
#         return prompt
#
#
# class TextWithForeignKeyPrompt(BasicPrompt):
#     template_info = "Given the following database schema:\n" \
#                     "{} \n" \
#                     "And their foreign keys:\n" \
#                     "{}"
#     template_question = "Answer the following: {}"
#
#     def format_question(self, example: dict):
#         schemas = "\n".join([f"{_.name}: {', '.join(_.schema)}" for _ in example["tables"]])
#         # foreign_keys
#         foreign_keys = list()
#         for table in example["tables"]:
#             for pair_str in table["table_info"]["foreign_key"]:
#                 a, b = [_.strip() for _ in pair_str[1:-1].split(",")]
#                 foreign_keys.append(f"{a}={b}")
#         foreign_keys = f"{', '.join(foreign_keys)}"
#
#         prompt_info = self.template_info.format(schemas, foreign_keys)
#         prompt_extra_info = self.get_extra_info(example["db_id"])
#         prompt_question = self.template_question.format(example["question"])
#
#         if prompt_extra_info is None or prompt_extra_info == "":
#             prompt_components = [prompt_info,prompt_question]
#         else:
#             prompt_components = [prompt_info, prompt_extra_info, prompt_question]
#
#         prompt = "\n".join(prompt_components)
#         return prompt
#
#
# class NumberSignWithForeignKeyPrompt(BasicPrompt):
#     template_info = "### Complete sqlite SQL query only and with no explanation\n" \
#                     "### SQLite SQL tables, with their properties:\n" \
#                     "#\n" \
#                     "{}\n" \
#                     "#\n" \
#                     "### Their foreign keys:\n" \
#                     "#\n" \
#                     "{}\n" \
#                     "#"
#     template_question = "### {}"
#
#     def format_question(self, example: dict):
#         schemas = "\n".join([f"# {_.name}({', '.join(_.schema)})" for _ in example["tables"]])
#         # foreign_keys
#         foreign_keys = list()
#         for table in example["tables"]:
#             for pair_str in table["table_info"]["foreign_key"]:
#                 a, b = [_.strip() for _ in pair_str[1:-1].split(",")]
#                 foreign_keys.append(f"{a}={b}")
#         foreign_keys = f"# Foreign_keys=({', '.join(foreign_keys)})"
#
#         prompt_info = self.template_info.format(schemas, foreign_keys)
#         prompt_extra_info = self.get_extra_info(example["db_id"])
#         prompt_question = self.template_question.format(example["question"])
#
#         if prompt_extra_info is None or prompt_extra_info == "":
#             prompt_components = [prompt_info, prompt_question]
#         else:
#             prompt_components = [prompt_info, prompt_extra_info, prompt_question]
#
#         prompt = "\n".join(prompt_components)
#         return prompt
#
#
# class BaselineWithoutForeignKeyPrompt(BasicPrompt):
#     template_info = "{}\n"
#     template_question = "Q: \"{}\""
#
#     def format_question(self, example: dict):
#         # schemas
#         schemas = "\n".join([f"Table {_.name}, columns = {_.schema}" for _ in example["tables"]]).replace("'", "")
#
#         # format prompt
#         prompt_info = self.template_info.format(schemas)
#         prompt_extra_info = self.get_extra_info(example["db_id"])
#         prompt_question = self.template_question.format(example["question"])
#
#         if prompt_extra_info is None or prompt_extra_info == "":
#             prompt_components = [prompt_info,prompt_question]
#         else:
#             prompt_components = [prompt_info, prompt_extra_info, prompt_question]
#
#         prompt = "".join(prompt_components)
#         return prompt
#
#     def format_target(self, example: dict):
#         return self.format_question(example) + "\nA: SELECT "
#
#
# class InstructionWithForeignKeyPrompt(BasicPrompt):
#     template_info = (
#         "Below is an instruction that describes a task, paired with an input that provides further context. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\nWrite a sql to answer the question \"{}\"\n\n### Input:\n{}\nForeign Keys:{}\n"
#     )
#     template_question = "### Response:"
#
#     def format_question(self, example: dict):
#         schemas = "\n".join([f"{_.name}({', '.join(_.schema)})" for _ in example["tables"]])
#         # foreign_keys
#         foreign_keys = list()
#         for table in example["tables"]:
#             for pair_str in table["table_info"]["foreign_key"]:
#                 a, b = [_.strip() for _ in pair_str[1:-1].split(",")]
#                 foreign_keys.append(f"{a}={b}")
#         foreign_keys = f"{', '.join(foreign_keys)}"
#
#         prompt_info = self.template_info.format(example["question"], schemas, foreign_keys)
#         prompt_extra_info = self.get_extra_info(example["db_id"])
#         prompt_question = self.template_question
#
#         if prompt_extra_info is None or prompt_extra_info == "":
#             prompt_components = [prompt_info, prompt_question]
#         else:
#             # TODO: extra_info should be after info
#             prompt_components = [prompt_info, prompt_extra_info, prompt_question]
#
#         prompt = "\n".join(prompt_components)
#         return prompt
#
#
# class SQLWithRulePrompt(BasicPrompt):
#     template_info =   "/* Given the following database schema: */\n" \
#                       "{}"
#     template_question =  "/* Answer the following with no explanation: {} */"
#
#     def format_question(self, example: dict):
#         sqls = get_sql_for_database(example["path_db"])
#
#         prompt_info = self.template_info.format("\n\n".join(sqls))
#         prompt_extra_info = self.get_extra_info(example["db_id"])
#         prompt_question = self.template_question.format(example["question"])
#
#         if prompt_extra_info is None or prompt_extra_info == "":
#             prompt_components = [prompt_info, prompt_question]
#         else:
#             prompt_components = [prompt_info, prompt_extra_info, prompt_question]
#
#         prompt = "\n\n".join(prompt_components)
#         return prompt
#
#
# class TextWithRulePrompt(BasicPrompt):
#     template_info = "Given the following database schema:\n" \
#                   "{}"
#     template_question = "Answer the following with no explanation: {}"
#
#     def format_question(self, example: dict):
#         schemas = "\n".join([f"{_.name}: {', '.join(_.schema)}" for _ in example["tables"]])
#
#         prompt_info = self.template_info.format(schemas)
#         prompt_extra_info = self.get_extra_info(example["db_id"])
#         prompt_question = self.template_question.format(example["question"])
#
#         if prompt_extra_info is None or prompt_extra_info == "":
#             prompt_components = [prompt_info,prompt_question]
#         else:
#             prompt_components = [prompt_info, prompt_extra_info, prompt_question]
#
#         prompt = "\n".join(prompt_components)
#         return prompt
#
#
# class NumberSignWithoutRulePrompt(BasicPrompt):
#     template_info = "### Complete sqlite SQL query\n" \
#                     "### SQLite SQL tables, with their properties:\n" \
#                     "#\n" \
#                     "{}\n" \
#                     "#"
#     template_question = "### {}"
#
#     def format_question(self, example: dict):
#         schemas = "\n".join([f"# {_.name}({', '.join(_.schema)})" for _ in example["tables"]])
#
#         prompt_info = self.template_info.format(schemas)
#         prompt_extra_info = self.get_extra_info(example["db_id"])
#         prompt_question = self.template_question.format(example["question"])
#
#         if prompt_extra_info is None or prompt_extra_info == "":
#             prompt_components = [prompt_info,prompt_question]
#         else:
#             prompt_components = [prompt_info, prompt_extra_info, prompt_question]
#
#         prompt = "\n".join(prompt_components)
#         return prompt
#
#
# class InstructionWithRulePrompt(BasicPrompt):
#     template_info = (
#         "Below is an instruction that describes a task, paired with an input that provides further context. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\nWrite a sql only and with no explanation to answer the question \"{}\"\n\n### Input:\n{}\n"
#     )
#     template_question = "### Response:"
#
#     def format_question(self, example: dict):
#         schemas = "\n".join([f"{_.name}({', '.join(_.schema)})" for _ in example["tables"]])
#
#         prompt_info = self.template_info.format(example["question"], schemas)
#         prompt_extra_info = self.get_extra_info(example["db_id"])
#         prompt_question = self.template_question
#
#         if prompt_extra_info is None or prompt_extra_info == "":
#             prompt_components = [prompt_info, prompt_question]
#         else:
#             # TODO: extra_info should be after info
#             prompt_components = [prompt_info, prompt_extra_info, prompt_question]
#
#         prompt = "\n".join(prompt_components)
#         return prompt
#
#
# class SQLCOTPrompt(BasicPrompt):
#     template_info =   "/* Given the following database schema: */\n" \
#                       "{}"
#     template_question =  "/* Let's think step by step. Answer the following: {} */"
#
#     def format_question(self, example: dict):
#         sqls = get_sql_for_database(example["path_db"])
#
#         prompt_info = self.template_info.format("\n\n".join(sqls))
#         prompt_extra_info = self.get_extra_info(example["db_id"])
#         prompt_question = self.template_question.format(example["question"])
#
#         if prompt_extra_info is None or prompt_extra_info == "":
#             prompt_components = [prompt_info, prompt_question]
#         else:
#             prompt_components = [prompt_info, prompt_extra_info, prompt_question]
#
#         prompt = "\n\n".join(prompt_components)
#         return prompt
#
#     def format_target(self, example: dict):
#         return self.format_question(example)
#
#
# class TextCOTPrompt(BasicPrompt):
#     template_info = "Given the following database schema:\n" \
#                   "{}"
#     template_question = "Let's think step by step. Answer the following: {}"
#
#     def format_question(self, example: dict):
#         schemas = "\n".join([f"{_.name}: {', '.join(_.schema)}" for _ in example["tables"]])
#
#         prompt_info = self.template_info.format(schemas)
#         prompt_extra_info = self.get_extra_info(example["db_id"])
#         prompt_question = self.template_question.format(example["question"])
#
#         if prompt_extra_info is None or prompt_extra_info == "":
#             prompt_components = [prompt_info,prompt_question]
#         else:
#             prompt_components = [prompt_info, prompt_extra_info, prompt_question]
#
#         prompt = "\n".join(prompt_components)
#         return prompt
#
#     def format_target(self, example: dict):
#         return self.format_question(example)
#
#
# class NumberSignCOTPrompt(BasicPrompt):
#     template_info = "### Let's think step by step. Complete sqlite SQL query only and with no explanation\n" \
#                     "### SQLite SQL tables, with their properties:\n" \
#                     "#\n" \
#                     "{}\n" \
#                     "#"
#     template_question = "### {}"
#
#     def format_question(self, example: dict):
#         schemas = "\n".join([f"# {_.name}({', '.join(_.schema)})" for _ in example["tables"]])
#
#         prompt_info = self.template_info.format(schemas)
#         prompt_extra_info = self.get_extra_info(example["db_id"])
#         prompt_question = self.template_question.format(example["question"])
#
#         if prompt_extra_info is None or prompt_extra_info == "":
#             prompt_components = [prompt_info,prompt_question]
#         else:
#             prompt_components = [prompt_info, prompt_extra_info, prompt_question]
#
#         prompt = "\n".join(prompt_components)
#         return prompt
#
#     def format_target(self, example: dict):
#         return self.format_question(example)
#
#
# class InstructionCOTPrompt(BasicPrompt):
#     template_info = (
#         "Below is an instruction that describes a task, paired with an input that provides further context. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\nLet's think step by step. Write a sql to answer the question \"{}\"\n\n### Input:\n{}\n"
#     )
#     template_question = "### Response:"
#
#     def format_question(self, example: dict):
#         schemas = "\n".join([f"{_.name}({', '.join(_.schema)})" for _ in example["tables"]])
#
#         prompt_info = self.template_info.format(example["question"], schemas)
#         prompt_extra_info = self.get_extra_info(example["db_id"])
#         prompt_question = self.template_question
#
#         if prompt_extra_info is None or prompt_extra_info == "":
#             prompt_components = [prompt_info, prompt_question]
#         else:
#             # TODO: extra_info should be after info
#             prompt_components = [prompt_info, prompt_extra_info, prompt_question]
#
#         prompt = "\n".join(prompt_components)
#         return prompt
#
#     def format_target(self, example: dict):
#         return self.format_question(example)
#
#
# class CBRPrompt(BasicPrompt):
#     template_info = "# The following are the table names and column names needed to generate SQL:\n" \
#                     "Tables: {}\n" \
#                     "Columns: *, {}\n" \
#                     "Foreign keys: {}"
#     template_question = '# translate "{}" into SQL query only and with no explanation:'
#
#     def format_question(self, example: dict):
#         tables = ", ".join([f"{_.name}" for _ in example["tables"]])
#         columns = ", ".join([f"{_.name}.{col}" for _ in example["tables"] for col in _.schema])
#         # foreign_keys
#         foreign_keys = list()
#         for table in example["tables"]:
#             for pair_str in table["table_info"]["foreign_key"]:
#                 a, b = [_.strip() for _ in pair_str[1:-1].split(",")]
#                 foreign_keys.append(f"{a}={b}")
#         foreign_keys = f"{', '.join(foreign_keys)}"
#
#         prompt_info = self.template_info.format(tables, columns, foreign_keys)
#         prompt_extra_info = self.get_extra_info(example["db_id"])
#         prompt_question = self.template_question.format(example["question"])
#
#         if prompt_extra_info is None or prompt_extra_info == "":
#             prompt_components = [prompt_info,prompt_question]
#         else:
#             prompt_components = [prompt_info, prompt_extra_info, prompt_question]
#
#         prompt = "\n".join(prompt_components)
#         return prompt
