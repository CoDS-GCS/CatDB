import numpy as np
import pandas as pd
# from sklearn.base import TransformerMixin
# from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder, OrdinalEncoder
# from .Cache import lazy_property

# import logging
# from typing import List, Union
# import scipy.sparse as sp
#
# log = logging.getLogger(__name__)
# AM = Union[np.ndarray, sp.spmatrix]
# DF = pd.DataFrame
#
#
# def _attributes(obj, filtr='all'):
#     attrs = vars(obj)
#     if filtr is None or filtr == 'all':
#         return attrs
#     elif filtr == 'public':
#         return {k: v for k, v in attrs.items() if not k.startswith('_')}
#     elif filtr == 'private':
#         return {k: v for k, v in attrs.items() if k.startswith('_')}
#     elif isinstance(filtr, list):
#         return {k: v for k, v in attrs.items() if k in filtr}
#     else:
#         assert callable(filtr)
#         return {k: v for k, v in attrs.items() if filtr(k)}
#
#
# def _classname(obj):
#     return type(obj).__name__
#
#
# def repr_def(obj, attributes='public'):
#     return "{cls}({attrs!r})".format(
#         cls=_classname(obj),
#         attrs=_attributes(obj, attributes)
#     )


class Dataset(object):
    def __init__(self,
                 dataset_name: str,
                 train_path: str,
                 test_path: str,
                 task_type: str,
                 target_attribute: str):
        self.dataset_name = dataset_name
        self.train_path = train_path
        self.test_path = test_path
        self.task_type = task_type
        self.target_attribute = target_attribute
