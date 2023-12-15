from src.main.python.catalog import ProfileInfo
import numpy as np
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_bool_dtype

class Profile(object):
    def __init__(self, data, nrows, columns):
        self.data = data
        self.nrows = nrows
        self.cols_profile = dict()
        self.categorical_ratio = int(0.5 * nrows)

        for col_name in columns:
            pro_info = ProfileInfo.ProfileInfo()
            pro_info.data_type = self.data.dtypes[col_name]
            pro_info.count, pro_info.number_of_nulls, pro_info.nullable = self.compute_counts(col_name=col_name)
            pro_info.is_categorical, pro_info.category_values, pro_info.category_values_len, pro_info.is_unique = self.compute_groups(col_name=col_name)
            pro_info.min_value, pro_info.max_value, pro_info.mean_value, pro_info.std_value = self.compute_statistic(col_name=col_name)
            pro_info.freq_value = self.data[col_name].value_counts()[:1].index.tolist()[0]
            self.cols_profile[col_name] = pro_info

    def compute_counts(self, col_name):
        number_of_nulls = self.data[col_name].isna().sum()
        nullable = number_of_nulls > 0
        return self.nrows - number_of_nulls, number_of_nulls, nullable

    def compute_groups(self, col_name):
        unique_values = self.data[col_name].unique()
        unique_len = len(unique_values)
        is_unique = unique_len == self.nrows
        is_categorical = unique_len <= self.categorical_ratio

        if is_categorical:
            return is_categorical, unique_values, unique_len, is_unique
        else:
            return is_categorical,[], 0, is_unique

    def compute_statistic(self, col_name):
        if is_numeric_dtype(self.data[col_name]) or is_bool_dtype(self.data[col_name]):
            col_min = np.min(self.data[col_name])
            col_max = np.max(self.data[col_name])
            col_mean = np.mean(self.data[col_name])
            col_std = np.std(self.data[col_name])
            return col_min, col_max, col_mean, col_std
        else:
            return None, None, None, None
