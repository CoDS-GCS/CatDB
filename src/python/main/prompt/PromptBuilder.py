from catalog.Catalog import CatalogInfo
from util.Config import PROMPT_FUNC, PROMPT_FUNC_MULTI_TABLE
from .ErrorPromptTemplate import RuntimeErrorPrompt, ResultsErrorPrompt, SyntaxErrorPrompt
from .PromptTemplateDataClean import CatDBCategoricalDataCleanPrompt
from .PromptTemplateCatalogClean import CatDBCatalogCleanPrompt


def get_representation_class(repr_type: str):
    representation_class = PROMPT_FUNC[repr_type]
    if representation_class is None:
        raise ValueError(f"{repr_type} is not supported yet")
    return representation_class


def get_representation_class_multi_table(repr_type: str):
    representation_class = PROMPT_FUNC_MULTI_TABLE[repr_type]
    if representation_class is None:
        raise ValueError(f"{repr_type} is not supported yet")
    return representation_class


def prompt_factory(catalog: [],
                   representation_type: str,
                   samples_type: str,
                   number_samples: int,
                   task_type: str,
                   number_iteration: int,
                   target_attribute: str,
                   data_source_train_path: str,
                   data_source_test_path: str,
                   dataset_description: str,
                   previous_result: str,
                   target_table: str,
                   dependency: dict()):
    cat = None
    file_format = None
    if len(catalog) == 1:
        repr_cls = get_representation_class(representation_type)
        cat = catalog[0]
        file_format = cat.file_format
    else:
        repr_cls = get_representation_class_multi_table(representation_type)
        file_format = catalog[0].file_format
        cat = catalog

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
            self.catalog = cat
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
            self.target_table = target_table
            self.dependency = dependency
            repr_cls.__init__(self, *args, **kwargs)

    return PromptClass()


def prompt_factory_data_cleaning(catalog: []):
    cat = None
    repr_cls = CatDBCategoricalDataCleanPrompt
    cat = catalog[0]
    class_name = f"CatDB-Data-Cleaning"
    assert repr_cls is not None

    class PromptClass(repr_cls):
        def __init__(self, *args, **kwargs):
            self.class_name = class_name
            self.catalog = cat
            self.number_samples = 0
            repr_cls.__init__(self, *args, **kwargs)

    return PromptClass()


def prompt_factory_data_catalog_cleaning(catalog: []):
    cat = None
    repr_cls = CatDBCatalogCleanPrompt
    cat = catalog[0]
    class_name = f"CatDB-Data-Catalog-Cleaning"
    assert repr_cls is not None

    class PromptClass(repr_cls):
        def __init__(self, *args, **kwargs):
            self.class_name = class_name
            self.catalog = cat
            self.number_samples = 100
            repr_cls.__init__(self, *args, **kwargs)

    return PromptClass()


def prompt_factory_missing_values(catalog: CatalogInfo,
                                  representation_type: str,
                                  number_samples: int,
                                  samples_missed_values,
                                  columns_has_missing_values,
                                  dataset_description: str,
                                  target_attribute: str,
                                  target_samples: str,
                                  target_samples_size: int
                                  ):
    repr_cls = get_representation_class(f'{representation_type}MissingValue')
    class_name = f"{representation_type}-MissingValueImputation-{number_samples}-SHOT"
    assert repr_cls is not None

    class PromptClass(repr_cls):
        def __init__(self, *args, **kwargs):
            self.class_name = class_name
            self.catalog = catalog
            self.number_samples = 0
            self.samples_missed_values = samples_missed_values
            self.missed_columns = columns_has_missing_values
            self.dataset_description = dataset_description
            self.target_attribute = target_attribute
            self.target_samples = target_samples
            self.target_samples_size = target_samples_size
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


def error_prompt_factory(pipeline_code: str, pipeline_error_class: str, pipeline_error_detail: str, schema_data: str,
                         task_type: str, data_source_train_path: str, data_source_test_path: str,
                         pipeline_type: str = None):
    if pipeline_error_class in {'NameError', 'InvalidIndexError'} or pipeline_type == "data-cleaning":
        error_prompt = SyntaxErrorPrompt(pipeline_code=pipeline_code,
                                         pipeline_error=f"{pipeline_error_class}: {pipeline_error_detail}").format_prompt()

    else:
        evaluation_text = get_evaluation_text(task_type=task_type)
        error_prompt = RuntimeErrorPrompt(pipeline_code=pipeline_code,
                                          pipeline_error=f"{pipeline_error_class}: {pipeline_error_detail}",
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
