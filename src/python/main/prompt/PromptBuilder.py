from catalog.Catalog import CatalogInfo
from util.Config import PROMPT_FUNC
from .ErrorPromptTemplate import RuntimeErrorPrompt, ResultsErrorPrompt, SyntaxErrorPrompt


def get_representation_class(repr_type: str):
    representation_class = PROMPT_FUNC[repr_type]
    if representation_class is None:
        raise ValueError(f"{repr_type} is not supported yet")
    return representation_class


def prompt_factory(catalog: CatalogInfo,
                   representation_type: str,
                   samples_type: str,
                   number_samples: int,
                   task_type: str,
                   number_iteration: int,
                   target_attribute: str,
                   data_source_train_path: str,
                   data_source_test_path: str,
                   dataset_description: str,
                   previous_result: str):
    repr_cls = get_representation_class(representation_type)
    file_format = catalog.file_format
    evaluation_text = None
    evaluation_text = get_evaluation_text(task_type=task_type)
    if task_type == "binary" or task_type == "multiclass":
        task_type_str = f"{task_type} classification"
    else:
        task_type_str = task_type

    class_name = f"{representation_type}-{samples_type}-{number_samples}-SHOT"
    assert repr_cls is not None

    class PromptClass(repr_cls):
        def __init__(self, *args, **kwargs):
            self.class_name = class_name
            self.catalog = catalog
            self.data_source_train_path = data_source_train_path
            self.data_source_test_path = data_source_test_path
            self.file_format = file_format
            self.number_samples = number_samples
            self.iterative = number_iteration
            self.target_attribute = target_attribute
            self.task_type = task_type_str
            self.evaluation_text = evaluation_text
            self.dataset_description = dataset_description
            self.previous_result = previous_result
            repr_cls.__init__(self, *args, **kwargs)

    return PromptClass()


def get_evaluation_text(task_type: str):
    from util.Config import _CODE_FORMATTING_BINARY_EVALUATION
    from util.Config import _CODE_FORMATTING_MULTICLASS_EVALUATION
    from util.Config import _CODE_FORMATTING_REGRESSION_EVALUATION

    if task_type == "binary" or task_type == "multiclass":
        if task_type == "binary":
            return _CODE_FORMATTING_BINARY_EVALUATION
        else:
            return _CODE_FORMATTING_MULTICLASS_EVALUATION
    else:
        return _CODE_FORMATTING_REGRESSION_EVALUATION


def error_prompt_factory(pipeline_code: str, pipeline_error_class: str, pipeline_error_detail: str, schema_data: str, task_type: str,
                         data_source_train_path: str, data_source_test_path: str):
    if pipeline_error_class in {'NameError','InvalidIndexError'}:
        error_prompt = SyntaxErrorPrompt(pipeline_code=pipeline_code, pipeline_error=f"{pipeline_error_class}: {pipeline_error_detail}").format_prompt()

    else:
        evaluation_text = get_evaluation_text(task_type=task_type)
        error_prompt = RuntimeErrorPrompt(pipeline_code=pipeline_code, pipeline_error=f"{pipeline_error_class}: {pipeline_error_detail}",
                                          schema_data=schema_data, evaluation_text=evaluation_text,
                                          data_source_train_path=data_source_train_path,
                                          data_source_test_path=data_source_test_path).format_prompt()
    return error_prompt['system_message'], error_prompt['user_message']


def result_error_prompt_factory(pipeline_code: str, task_type: str,
                                data_source_train_path: str, data_source_test_path: str):
    evaluation_text = get_evaluation_text(task_type=task_type)
    error_prompt = ResultsErrorPrompt(pipeline_code=pipeline_code, evaluation_text=evaluation_text,
                                      data_source_train_path=data_source_train_path,
                                      data_source_test_path=data_source_test_path).format_prompt()
    return error_prompt['system_message'], error_prompt['user_message']
