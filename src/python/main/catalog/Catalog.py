import os
from .Profile import load_JSON_profile_info, load_JSON_profile_info_with_update
from .DimensionReduction import ReduceDimension
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


def load_data_source_profile(data_source_path: str, file_format: str, target_attribute: str, enable_reduction: bool,
                             dependency: Dependency=None, categorical_values_restricted_size: int=50, cleaning: bool=False):
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
            if cleaning:
                profile = load_JSON_profile_info(f'{data_source_path}/{d}/{f}',
                                                 categorical_values_restricted_size=categorical_values_restricted_size)
            else:
                profile = load_JSON_profile_info_with_update( data_profile_update=f'{data_source_path}_update',
                                                 file_name=f'{data_source_path}/{d}/{f}')
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

    orig_profile_size = len(schema_info)
    drop_schema_info = dict()
    if enable_reduction:
        rd = ReduceDimension(profile_info=profile_info, target_attribute=target_attribute)

        schema_info, schema_info_group, drop_schema_info, profile_info = rd.get_new_profile_info()
        new_profile_size = len(schema_info)

        for dk in drop_schema_info.keys():
            if dk in columns_categorical:
                columns_categorical.remove(dk)
                if dk in columns_categorical_missing_values:
                    columns_categorical_missing_values.remove(dk)

            if dk in columns_numerical:
                columns_numerical.remove(dk)
                if dk in columns_numerical_missing_values:
                    columns_numerical_missing_values.remove(dk)

            if dk in columns_bool:
                columns_bool.remove(dk)
                if dk in columns_bool_missing_values:
                    columns_bool_missing_values.remove(dk)

            if dk in columns_others:
                columns_others .remove(dk)
                if dk in columns_others_missing_values:
                    columns_others_missing_values.remove(dk)

        # print(
        #     f"[{data_source_path}]  --- orig_size = {orig_profile_size}, new_size = {new_profile_size} >> r= {orig_profile_size - new_profile_size}")

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


def load_data_source_profile_as_chunck(data_source_path: str, file_format: str, target_attribute: str, enable_reduction: bool,
                             dependency: Dependency=None, categorical_values_restricted_size: int=50, cleaning: bool=False, chunk_size:int=1):
    profile_info = dict()
    schema_info = dict()
    ncols = 0
    nrows = 0
    dataset_name = None
    source_path = None
    schema_info_group = dict()
    col_group_name = dict()

    columns_categorical = []
    columns_categorical_missing_values = []
    columns_numerical = []
    columns_numerical_missing_values = []
    columns_bool = []
    columns_bool_missing_values = []
    columns_others = []
    columns_others_missing_values = []
    all_ols = []
    table_name = None

    for d in os.listdir(data_source_path):
        files = [f for f in os.listdir(f'{data_source_path}/{d}/')]
        for f in files:
            if cleaning:
                profile = load_JSON_profile_info(f'{data_source_path}/{d}/{f}',
                                                 categorical_values_restricted_size=categorical_values_restricted_size)
            else:
                profile = load_JSON_profile_info_with_update(data_profile_update=f'{data_source_path}_update',
                                                             file_name=f'{data_source_path}/{d}/{f}')
            table_name = profile.table_name
            profile_info[profile.column_name] = profile
            schema_info[profile.column_name] = profile.short_data_type

            if schema_info_group.get(profile.short_data_type) is None:
                schema_info_group[profile.short_data_type] = []
            else:
                schema_info_group[profile.short_data_type].append(profile.column_name)
                col_group_name[profile.column_name] = profile.short_data_type

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

    orig_profile_size = len(schema_info)
    drop_schema_info = dict()
    if enable_reduction:
        rd = ReduceDimension(profile_info=profile_info, target_attribute=target_attribute)

        schema_info, schema_info_group, drop_schema_info, profile_info = rd.get_new_profile_info()
        new_profile_size = len(schema_info)

        for dk in drop_schema_info.keys():
            if dk in columns_categorical:
                columns_categorical.remove(dk)
                if dk in columns_categorical_missing_values:
                    columns_categorical_missing_values.remove(dk)

            if dk in columns_numerical:
                columns_numerical.remove(dk)
                if dk in columns_numerical_missing_values:
                    columns_numerical_missing_values.remove(dk)

            if dk in columns_bool:
                columns_bool.remove(dk)
                if dk in columns_bool_missing_values:
                    columns_bool_missing_values.remove(dk)

            if dk in columns_others:
                columns_others.remove(dk)
                if dk in columns_others_missing_values:
                    columns_others_missing_values.remove(dk)

    all_ols.extend(columns_categorical)
    all_ols.extend(columns_numerical)
    all_ols.extend(columns_bool)
    all_ols.extend(columns_others)

    chuncks = [all_ols[i:i + chunk_size] for i in range(0, len(all_ols), chunk_size)]
    catalog_chuncks = []

    for ch in chuncks:
        profile_info_ch = dict()
        schema_info_ch = dict()
        ncols = len(ch)
        schema_info_group_ch = dict()
        drop_schema_info_ch = dict()

        columns_categorical_ch = []
        columns_categorical_missing_values_ch = []
        columns_numerical_ch = []
        columns_numerical_missing_values_ch = []
        columns_bool_ch = []
        columns_bool_missing_values_ch = []
        columns_others_ch = []
        columns_others_missing_values_ch = []
        for col in ch:
            schema_info_ch[col] = schema_info[col]
            profile_info_ch[col] = profile_info[col]
            schema_info_group_ch[col_group_name[col]] = col
            if col in drop_schema_info.keys():
                drop_schema_info_ch[col] = drop_schema_info[col]

            if col in columns_categorical:
                columns_categorical_ch.append(col)
                if col in columns_categorical_missing_values:
                    columns_categorical_missing_values_ch.append(ch)
            elif col in columns_numerical:
                columns_numerical_ch.append(col)
                if col in columns_numerical_missing_values:
                    columns_numerical_missing_values_ch.append(col)
            elif col in columns_bool:
                columns_bool_ch.append(col)
                if col in columns_bool_missing_values:
                    columns_bool_missing_values_ch.append(col)
            elif col in columns_others:
                columns_others_ch.append(col)
                if col in columns_others_missing_values:
                    columns_others_missing_values_ch.append(col)

            catalog_chuncks.append(CatalogInfo(nrows=nrows,
                           ncols=ncols,
                           file_format="csv",
                           dataset_name=dataset_name,
                           table_name=table_name,
                           schema_info=schema_info,
                           profile_info=profile_info,
                           data_source_path=source_path,
                           drop_schema_info=drop_schema_info_ch,
                           schema_info_group=schema_info_group_ch,
                           columns_categorical=columns_categorica_chl,
                           columns_categorical_missing_values=columns_categorical_missing_values_ch,
                           columns_numerical=columns_numerical_ch,
                           columns_numerical_missing_values=columns_numerical_missing_values_ch,
                           columns_bool=columns_bool_ch,
                           columns_bool_missing_values=columns_bool_missing_values_ch,
                           columns_others=columns_others_ch,
                           columns_others_missing_values=columns_others_missing_values_ch,
                           dependency=dependency
                           ))

        return catalog_chuncks