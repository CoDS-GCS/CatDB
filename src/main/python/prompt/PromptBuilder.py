from .BasicICLPrompt import BasicICLPrompt
from catalog.Catalog import CatalogInfo
from util.StaticValues import REPRESENTATION_TYPE
from .PromptTemplate import *


def get_representation_class(repr_type: str):
    if repr_type == REPRESENTATION_TYPE.TEXT:
        representation_class = TextPrompt
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
                   data_source_train_path:str,
                   data_source_test_path: str):

    repr_cls = get_representation_class(representation_type)
    schema_info = catalog.schema_info
    profile_info = catalog.profile_info
    file_format = catalog.file_format

    if number_example == 0:
        assert repr_cls is not None
        class_name = f"{representation_type}-{example_type}-{number_example}-SHOT"

        class PromptClass(repr_cls, BasicICLPrompt):
            def __init__(self, *args, **kwargs):
                self.class_name = class_name
                self.schema = schema_info
                self.profile = profile_info
                self.data_source_train_path = data_source_train_path
                self.data_source_test_path = data_source_test_path
                self.file_format = file_format
                self.number_example = number_example
                self.iterative = number_iteration
                self.target_attribute = target_attribute
                self.task_type = task_type
                repr_cls.__init__(self, *args, **kwargs)
                BasicICLPrompt.__init__(self, *args, **kwargs)

        return PromptClass()
