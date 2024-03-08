import os
from .Profile import load_JSON_profile_info
from .DimensionReduction import ReduceDimension


class CatalogInfo(object):
    def __init__(self, nrows: int, ncols: int, dataset_name: str, data_source_path, file_format: str, schema_info: dict,
                 drop_schema_info: dict, profile_info: dict, schema_info_group: dict):
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


def load_data_source_profile(data_source_path: str, file_format: str, target_attribute: str, enable_reduction: bool):
    profile_info = dict()
    schema_info = dict()
    ncols = 0
    nrows = 0
    dataset_name = None
    source_path = None
    schema_info_group = dict()

    for d in os.listdir(data_source_path):
        files = [f for f in os.listdir(f'{data_source_path}/{d}/')]
        for f in files:
            profile = load_JSON_profile_info(f'{data_source_path}/{d}/{f}')
            profile_info[profile.column_name] = profile
            schema_info[profile.column_name] = profile.short_data_type

            if schema_info_group.get(profile.short_data_type) is None:
                schema_info_group[profile.short_data_type] = []
            else:
                schema_info_group[profile.short_data_type].append(profile.column_name)

            ncols += 1
            nrows = max(profile.total_values_count, nrows)
            if dataset_name is None:
                dataset_name = profile.dataset_name
                source_path = profile.path

    orig_profile_size = len(schema_info)
    drop_schema_info = dict()
    if enable_reduction:
        rd = ReduceDimension(profile_info=profile_info, target_attribute=target_attribute)

        schema_info, schema_info_group, drop_schema_info, profile_info = rd.get_new_profile_info()
        new_profile_size = len(schema_info)

        print(
            f"[{data_source_path}]  --- orig_size = {orig_profile_size}, new_size = {new_profile_size} >> r= {orig_profile_size - new_profile_size}")

    return CatalogInfo(nrows=nrows, ncols=ncols, file_format="csv", dataset_name=dataset_name,
                       schema_info=schema_info, profile_info=profile_info, data_source_path=source_path,
                       drop_schema_info=drop_schema_info, schema_info_group=schema_info_group)
