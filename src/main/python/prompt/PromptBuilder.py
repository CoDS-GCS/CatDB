from src.main.python.catalog.Catalog import CatalogInfo
from src.main.python.prompt.BasicICLPrompt import BasicICLPrompt
from src.main.python.util.StaticValues import REPRESENTATION_TYPE
from src.main.python.prompt.PromptTemplate import *
#--------------------------------
# from utils.enums import REPR_TYPE
# from utils.enums import EXAMPLE_TYPE
# from utils.enums import SELECTOR_TYPE
# from prompt.PromptReprTemplate import *
# from prompt.ExampleFormatTemplate import *
# from prompt.ExampleSelectorTemplate import *
# from prompt.PromptICLTemplate import BasicICLPrompt


def get_representation_class(repr_type: str):
    if repr_type == REPRESENTATION_TYPE.TEXT:
        representation_class = TextPrompt
    else:
        raise ValueError(f"{repr_type} is not supproted yet")
    return representation_class


# def get_example_format_cls(example_format: str):
#     if example_format == EXAMPLE_TYPE.ONLY_SQL:
#         example_format_cls = SqlExampleStyle
#     elif example_format == EXAMPLE_TYPE.QA:
#         example_format_cls = QuestionSqlExampleStyle
#     elif example_format == EXAMPLE_TYPE.QAWRULE:
#         example_format_cls = QuestionSqlWithRuleExampleStyle
#     elif example_format == EXAMPLE_TYPE.COMPLETE:
#         example_format_cls = CompleteExampleStyle
#     elif example_format == EXAMPLE_TYPE.NUMBER_SIGN_QA:
#         example_format_cls = NumberSignQuestionSqlExampleStyle
#     elif example_format == EXAMPLE_TYPE.BASELINE_QA:
#         example_format_cls = BaselineQuestionSqlExampleStyle
#     else:
#         raise ValueError(f"{example_format} is not supported yet!")
#     return example_format_cls
#
#
# def get_example_selector(selector_type: str):
#     if selector_type == SELECTOR_TYPE.COS_SIMILAR:
#         selector_cls = CosineSimilarExampleSelector
#     elif selector_type == SELECTOR_TYPE.RANDOM:
#         selector_cls = RandomExampleSelector
#     elif selector_type == SELECTOR_TYPE.EUC_DISTANCE:
#         selector_cls = EuclideanDistanceExampleSelector
#     elif selector_type == SELECTOR_TYPE.EUC_DISTANCE_THRESHOLD:
#         selector_cls = EuclideanDistanceThresholdExampleSelector
#     elif selector_type == SELECTOR_TYPE.EUC_DISTANCE_SKELETON_SIMILARITY_THRESHOLD:
#         selector_cls = EuclideanDistanceSkeletonSimilarThresholdSelector
#     elif selector_type == SELECTOR_TYPE.EUC_DISTANCE_QUESTION_MASK:
#         selector_cls = EuclideanDistanceQuestionMaskSelector
#     elif selector_type == SELECTOR_TYPE.EUC_DISTANCE_PRE_SKELETON_SIMILARITY_THRESHOLD:
#         selector_cls = EuclideanDistancePreSkeletonSimilarThresholdSelector
#     elif selector_type == SELECTOR_TYPE.EUC_DISTANCE_PRE_SKELETON_SIMILARITY_PLUS:
#         selector_cls = EuclideanDistancePreSkeletonSimilarPlusSelector
#     elif selector_type == SELECTOR_TYPE.EUC_DISTANCE_MASK_PRE_SKELETON_SIMILARITY_THRESHOLD:
#         selector_cls = EuclideanDistanceQuestionMaskPreSkeletonSimilarThresholdSelector
#     elif selector_type == SELECTOR_TYPE.EUC_DISTANCE_MASK_PRE_SKELETON_SIMILARITY_THRESHOLD_SHIFT:
#         selector_cls = EuclideanDistanceQuestionMaskPreSkeletonSimilarThresholdShiftSelector
#     else:
#         raise ValueError(f"{selector_type} is not supported yet!")
#     return selector_cls


def prompt_factory(catalog: CatalogInfo, repr_type: str, k_shot: int, iterative:int, target_attribute:str):
    repr_cls = get_representation_class(repr_type)

    schema_info = catalog.schema_info
    profile_info = catalog.profile_info

    if k_shot == 0:
        assert repr_cls is not None
        class_name = f"{repr_type}_{k_shot}-SHOT"

        class PromptClass(repr_cls, BasicICLPrompt):
            def __init__(self, *args, **kwargs):
                self.class_name = class_name
                self.schema = schema_info
                self.profile = profile_info
                self.number_example = k_shot
                self.iterative = iterative
                self.target_attribute = target_attribute
                repr_cls.__init__(self,*args, **kwargs)
                BasicICLPrompt.__init__(self, *args, **kwargs)

        return PromptClass()
