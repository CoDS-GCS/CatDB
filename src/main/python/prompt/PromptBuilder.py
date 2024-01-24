from .BasicICLPrompt import BasicICLPrompt
from catalog.Catalog import CatalogInfo
from util.StaticValues import REPRESENTATION_TYPE
from .PromptTemplate import *


def get_representation_class(repr_type: str):
    if repr_type == REPRESENTATION_TYPE.SCHEMA:
        representation_class = SchemaPrompt
    elif repr_type == REPRESENTATION_TYPE.DISTINCT:
        representation_class = SchemaDistinctPrompt
    elif repr_type == REPRESENTATION_TYPE.MISSING_VALUE:
        representation_class = SchemaMissingValuePrompt
    elif repr_type == StaticValues.REPRESENTATION_TYPE.NUMERIC_STATISTIC:
        representation_class = SchemaNumericStatisticPrompt
    elif repr_type == REPRESENTATION_TYPE.CATEGORICAL_VALUE:
        representation_class = SchemaCategoricalValuePrompt

    elif repr_type == REPRESENTATION_TYPE.DISTINCT_MISSING_VALUE:
        representation_class = SchemaDistinctMissingValuePrompt
    elif repr_type == REPRESENTATION_TYPE.DISTINCT_NUMERIC_STATISTIC:
        representation_class = SchemaDistinctNumericStatisticPrompt
    elif repr_type == REPRESENTATION_TYPE.MISSING_VALUE_NUMERIC_STATISTIC:
        representation_class = SchemaMissingValueNumericStatisticPrompt
    elif repr_type == REPRESENTATION_TYPE.MISSING_VALUE_CATEGORICAL_VALUE:
        representation_class = SchemaMissingCategoricalValuePrompt
    elif repr_type == REPRESENTATION_TYPE.NUMERIC_STATISTIC_CATEGORICAL_VALUE:
        representation_class = SchemaNumericStatisticCategoricalValuePrompt
    else:
        raise ValueError(f"{repr_type} is not supported yet")
    return representation_class


def prompt_factory(catalog: CatalogInfo,
                   representation_type: str,
                   example_type: str,
                   number_example: str,
                   task_type: str,
                   number_iteration: int,
                   target_attribute: str,
                   data_source_train_path: str,
                   data_source_test_path: str,
                   suggested_model: str):
    repr_cls = get_representation_class(representation_type)
    schema_info = catalog.schema_info
    profile_info = catalog.profile_info
    nrows = catalog.nrows
    file_format = catalog.file_format
    evaluation_text = None
    if task_type == "binary" or task_type == "multiclass":
        task_type_str = f"{task_type} classification"
        if task_type == "binary":
            evaluation_text = StaticValues.CODE_FORMATTING_BINARY_EVALUATION
        else:
            evaluation_text = StaticValues.CODE_FORMATTING_MULTICLASS_EVALUATION
    else:
        task_type_str = task_type
        evaluation_text = StaticValues.CODE_FORMATTING_REGRESSION_EVALUATION

    class_name = f"{representation_type}-{example_type}-{number_example}-SHOT"

    if number_example == 0:
        assert repr_cls is not None

        class PromptClass(repr_cls, BasicICLPrompt):
            def __init__(self, *args, **kwargs):
                self.class_name = class_name
                self.schema = schema_info
                self.profile = profile_info
                self.nrows = nrows
                self.data_source_train_path = data_source_train_path
                self.data_source_test_path = data_source_test_path
                self.file_format = file_format
                self.number_example = number_example
                self.iterative = number_iteration
                self.target_attribute = target_attribute
                self.task_type = task_type_str
                self.evaluation_text = evaluation_text
                self.examples = None
                self.suggested_model = suggested_model
                repr_cls.__init__(self, *args, **kwargs)
                BasicICLPrompt.__init__(self, *args, **kwargs)

        return PromptClass()
