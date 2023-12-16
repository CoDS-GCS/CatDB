import numpy as np
from pandas import DataFrame
from src.main.python.util import StaticValues
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_bool_dtype
from pandas.api.types import is_string_dtype


class ProfileInfo(object):
    def __init__(self,
                 count=0,
                 nullable=False,
                 number_of_nulls=-1,
                 is_categorical=False,
                 category_values=None,
                 category_values_len=0,
                 min_value=None,
                 max_value=None,
                 mean_value=None,
                 std_value=None,
                 is_unique=False,
                 top_value=None,
                 freq_value=None):
        self.category_values_len = category_values_len
        if category_values is None:
            category_values = []
        self.freq_value = freq_value
        self.top_value = top_value
        self.is_unique = is_unique
        self.std_value = std_value
        self.mean_value = mean_value
        self.max_value = max_value
        self.min_value = min_value
        self.category_values = category_values
        self.is_categorical = is_categorical
        self.number_of_nulls = number_of_nulls
        self.nullable = nullable
        self.count = count


def get_schema_info(data: DataFrame):
    values = dict()
    for column_name in data.columns:
        is_str = is_string_dtype(data[column_name])
        if is_str:
            values[column_name] = 'string'
        elif is_numeric_dtype(data[column_name]):
            values[column_name] = data.dtypes[column_name].name
        else:
            values[column_name] = 'bool'

    return values


def get_profile_info(data: DataFrame, schema_info: dict):
    values = dict()
    nrows = len(data)
    for column_name in schema_info.keys():
        pro_info = ProfileInfo()
        pro_info.data_type = schema_info[column_name]
        pro_info.count, pro_info.number_of_nulls, pro_info.nullable = compute_counts(data=data, column_name=column_name, nrows=nrows)
        pro_info.is_categorical, pro_info.category_values, pro_info.category_values_len, pro_info.is_unique = compute_groups(data=data,
            column_name=column_name, nrows=nrows)
        pro_info.min_value, pro_info.max_value, pro_info.mean_value, pro_info.std_value = compute_statistic(data=data,
            column_name=column_name)
        pro_info.freq_value = data[column_name].value_counts()[:1].index.tolist()[0]
        values[column_name] = pro_info

    return values


def compute_counts(data: DataFrame, nrows: int, column_name: str):
    number_of_nulls = data[column_name].isna().sum()
    nullable = number_of_nulls > 0
    return nrows - number_of_nulls, number_of_nulls, nullable


def compute_groups(data: DataFrame, nrows: int, column_name: str):
    unique_values = data[column_name].unique()
    unique_len = len(unique_values)
    is_unique = unique_len == nrows
    is_categorical = unique_len <= StaticValues.CATEGORICAL_RATIO * nrows

    if is_categorical:
        return is_categorical, unique_values, unique_len, is_unique
    else:
        return is_categorical, [], 0, is_unique


def compute_statistic(data: DataFrame, column_name: str):
    if is_numeric_dtype(data[column_name]) or is_bool_dtype(data[column_name]):
        col_min = np.min(data[column_name])
        col_max = np.max(data[column_name])
        col_mean = np.mean(data[column_name])
        col_std = np.std(data[column_name])
        return col_min, col_max, col_mean, col_std
    else:
        return None, None, None, None
