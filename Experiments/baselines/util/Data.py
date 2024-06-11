from abc import ABC, abstractmethod
from enum import Enum, auto
import logging
from typing import List, Union

import numpy as np
import pandas as pd

# from .datautils import Encoder
# from .utils import clear_cache, lazy_property, profile, repr_def
#
# log = logging.getLogger(__name__)
#
#
# AM = Union[np.ndarray, sp.spmatrix]
# DF = pd.DataFrame


class Dataset(object):
    def __init__(self,
                 dataset_name: str,
                 train_path: str,
                 test_path:str,
                 task_type: str,
                 target_attribute: str):
        self.dataset_name = dataset_name
        self.train_path = train_path
        self.test_path = test_path
        self.task_type = task_type
        self.target_attribute = target_attribute


class Feature:

    def __init__(self, index, name, data_type, values=None, has_missing_values=False, is_target=False):
        """
        :param index: index of the feature in the full data frame.
        :param name: name of the feature.
        :param data_type: one of pandas-compatible type ('int', 'float', 'number', 'bool', 'category', 'string', 'object', 'datetime').
        :param values: for categorical features, the sorted list of accepted values.
        :param has_missing_values: True iff the feature has any missing values in the complete dataset.
        :param is_target: True for the target column.
        """
        self.index = index
        self.name = name
        self.data_type = data_type.lower() if data_type is not None else None
        self.values = values
        self.has_missing_values = has_missing_values
        self.is_target = is_target
        # print(self)

    def is_categorical(self, strict=True):
        if strict:
            return self.data_type == 'category'
        else:
            return self.data_type is not None and not self.is_numerical()

    def is_numerical(self):
        return self.data_type in ['int', 'float', 'number']

    @lazy_property
    def label_encoder(self):
        return Encoder('label' if self.values is not None else 'no-op',
                       target=self.is_target,
                       encoded_type=int if self.is_target and not self.is_numerical() else float,
                       missing_values=[None, np.nan, pd.NA],
                       missing_policy='mask' if self.has_missing_values else 'ignore',
                       normalize_fn=self.normalize
                       ).fit(self.values)

    @lazy_property
    def one_hot_encoder(self):
        return Encoder('one-hot' if self.values is not None else 'no-op',
                       target=self.is_target,
                       encoded_type=int if self.is_target and not self.is_numerical() else float,
                       missing_values=[None, np.nan, pd.NA],
                       missing_policy='mask' if self.has_missing_values else 'ignore',
                       normalize_fn=self.normalize
                       ).fit(self.values)

    def normalize(self, arr):
        return np.char.lower(np.char.strip(np.asarray(arr).astype(str)))

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        self._values = self.normalize(values).tolist() if values is not None else None

    def __repr__(self):
        return repr_def(self, 'all')