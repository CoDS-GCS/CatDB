from src.main.python.catalog.Catalog import CatalogInfo
from .BasicICLPrompt import BasicICLPrompt
from ..util.StaticValues import REPRESENTATION_TYPE
from .PromptTemplate import *


def get_representation_class(repr_type: str):
    if repr_type == REPRESENTATION_TYPE.TEXT:
        representation_class = TextPrompt
    else:
        raise ValueError(f"{repr_type} is not supported yet")
    return representation_class


def prompt_factory(catalog: CatalogInfo, repr_type: str, k_shot: int, iterative:int, target_attribute:str):
    repr_cls = get_representation_class(repr_type)

    schema_info = catalog.schema_info
    profile_info = catalog.profile_info
    task_type = "Binary Classification"
    file_format = catalog.file_format
    data_source_test_path = catalog.data_source_path

    if k_shot == 0:
        assert repr_cls is not None
        class_name = f"{repr_type}_{k_shot}-SHOT"

        class PromptClass(repr_cls, BasicICLPrompt):
            def __init__(self, *args, **kwargs):
                self.class_name = class_name
                self.schema = schema_info
                self.profile = profile_info
                self.data_source_train_path = catalog.data_source_path
                self.data_source_test_path = data_source_test_path
                self.file_format = file_format
                self.number_example = k_shot
                self.iterative = iterative
                self.target_attribute = target_attribute
                self.task_type = task_type
                repr_cls.__init__(self,*args, **kwargs)
                BasicICLPrompt.__init__(self, *args, **kwargs)

        return PromptClass()
