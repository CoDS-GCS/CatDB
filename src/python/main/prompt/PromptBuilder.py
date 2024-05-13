from catalog.Catalog import CatalogInfo
from util.StaticValues import CODE_FORMATTING_BINARY_EVALUATION
from util.StaticValues import CODE_FORMATTING_MULTICLASS_EVALUATION
from util.StaticValues import CODE_FORMATTING_REGRESSION_EVALUATION
from util.Config import PROMPT_FUNC
from ErrorPromptTemplate import RuntimeErrorPrompt


def get_representation_class(repr_type: str):
    representation_class = PROMPT_FUNC[repr_type]
    if representation_class is None:
        raise ValueError(f"{repr_type} is not supported yet")
    return representation_class


def prompt_factory(catalog: CatalogInfo,
                   representation_type: str,
                   sample_type: str,
                   number_samples: int,
                   task_type: str,
                   number_iteration: int,
                   target_attribute: str,
                   data_source_train_path: str,
                   data_source_test_path: str,
                   number_folds: int,
                   dataset_description: str):
    repr_cls = get_representation_class(representation_type)
    schema_info = catalog.schema_info
    profile_info = catalog.profile_info
    dropped_columns = catalog.drop_schema_info
    nrows = catalog.nrows
    file_format = catalog.file_format
    evaluation_text = None
    if task_type == "binary" or task_type == "multiclass":
        task_type_str = f"{task_type} classification"
        if task_type == "binary":
            evaluation_text = CODE_FORMATTING_BINARY_EVALUATION
        else:
            evaluation_text = CODE_FORMATTING_MULTICLASS_EVALUATION
    else:
        task_type_str = task_type
        evaluation_text = CODE_FORMATTING_REGRESSION_EVALUATION

    class_name = f"{representation_type}-{sample_type}-{number_samples}-SHOT"
    assert repr_cls is not None

    class PromptClass(repr_cls):
        def __init__(self, *args, **kwargs):
            self.class_name = class_name
            self.schema = schema_info
            self.profile = profile_info
            self.nrows = nrows
            self.data_source_train_path = data_source_train_path
            self.data_source_test_path = data_source_test_path
            self.file_format = file_format
            self.number_samples = number_samples
            self.iterative = number_iteration
            self.target_attribute = target_attribute
            self.task_type = task_type_str
            self.evaluation_text = evaluation_text
            self.examples = None
            self.dropped_columns = dropped_columns
            self.number_folds = number_folds
            self.dataset_description = dataset_description
            repr_cls.__init__(self, *args, **kwargs)

    return PromptClass()


def error_prompt_factory(pipeline_code: str, pipeline_error):
    error_prompt = RuntimeErrorPrompt(pipeline_code=pipeline_code, pipeline_error=pipeline_error)
    return error_prompt.format_rules(), error_prompt.format_question()
