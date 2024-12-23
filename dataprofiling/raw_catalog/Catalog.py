import os
from .Profile import load_JSON_profile_info
from .Dependency import Dependency


class CatalogInfo(object):
    def __init__(self,
                 nrows: int,
                 ncols: int,
                 dataset_name: str,
                 table_name: str,
                 data_source_path,
                 file_format: str,
                 schema_info: dict,
                 drop_schema_info: dict,
                 profile_info: dict,
                 schema_info_group: dict,
                 columns_categorical: list,
                 columns_categorical_missing_values: list,
                 columns_numerical: list,
                 columns_numerical_missing_values: list,
                 columns_bool: list,
                 columns_bool_missing_values: list,
                 columns_others: list,
                 columns_others_missing_values: list,
                 dependency: Dependency = None):
        self.profile_info = profile_info
        self.schema_info = schema_info
        self.file_format = file_format
        self.dataset_name = dataset_name
        self.data_source_path = data_source_path
        self.nrows = nrows
        self.ncols = ncols
        self.drop_schema_info = dict()
        if drop_schema_info is not None:
            self.ncols -= len(drop_schema_info)
            self.drop_schema_info = drop_schema_info
        self.schema_info_group = schema_info_group

        self.columns_categorical = columns_categorical
        self.columns_categorical_missing_values = columns_categorical_missing_values
        self.columns_numerical = columns_numerical
        self.columns_numerical_missing_values = columns_numerical_missing_values
        self.columns_bool = columns_bool
        self.columns_bool_missing_values = columns_bool_missing_values
        self.columns_others = columns_others
        self.columns_others_missing_values = columns_others_missing_values
        self.table_name = table_name
        self.dependency = dependency


def load_data_source_profile(data_source_path: str, dependency: Dependency=None):
    profile_info = dict()
    schema_info = dict()
    ncols = 0
    nrows = 0
    dataset_name = None
    source_path = None
    schema_info_group = dict()

    columns_categorical = []
    columns_categorical_missing_values = []
    columns_numerical = []
    columns_numerical_missing_values = []
    columns_bool = []
    columns_bool_missing_values = []
    columns_others = []
    columns_others_missing_values = []
    table_name = None

    for d in os.listdir(data_source_path):
        files = [f for f in os.listdir(f'{data_source_path}/{d}/')]
        for f in files:
            profile = load_JSON_profile_info(f'{data_source_path}/{d}/{f}', categorical_values_restricted_size=-1)
            table_name = profile.table_name
            profile_info[profile.column_name] = profile
            schema_info[profile.column_name] = profile.short_data_type

            if schema_info_group.get(profile.short_data_type) is None:
                schema_info_group[profile.short_data_type] = []
            else:
                schema_info_group[profile.short_data_type].append(profile.column_name)

            nrows = profile.nrows
            ncols += 1
            if dataset_name is None:
                dataset_name = profile.dataset_name
                source_path = profile.path

            if profile.short_data_type == 'list':
                continue

            if profile.is_categorical:
                columns_categorical.append(profile.column_name)
                if profile.missing_values_count > 0:
                    columns_categorical_missing_values.append(profile.column_name)

            elif profile.short_data_type in {"int", "float"}:
                columns_numerical.append(profile.column_name)
                if profile.missing_values_count > 0:
                    columns_numerical_missing_values.append(profile.column_name)

            elif profile.short_data_type == "bool":
                columns_bool.append(profile.column_name)
                if profile.missing_values_count > 0:
                    columns_bool_missing_values.append(profile.column_name)

            else:
                columns_others.append(profile.column_name)
                if profile.missing_values_count > 0:
                    columns_others_missing_values.append(profile.column_name)

    drop_schema_info = dict()
    return CatalogInfo(nrows=nrows,
                       ncols=ncols,
                       file_format="csv",
                       dataset_name=dataset_name,
                       table_name=table_name,
                       schema_info=schema_info,
                       profile_info=profile_info,
                       data_source_path=source_path,
                       drop_schema_info=drop_schema_info,
                       schema_info_group=schema_info_group,
                       columns_categorical=columns_categorical,
                       columns_categorical_missing_values=columns_categorical_missing_values,
                       columns_numerical=columns_numerical,
                       columns_numerical_missing_values=columns_numerical_missing_values,
                       columns_bool=columns_bool,
                       columns_bool_missing_values=columns_bool_missing_values,
                       columns_others=columns_others,
                       columns_others_missing_values=columns_others_missing_values,
                       dependency=dependency
                       )
