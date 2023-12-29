import os
from .Profile import load_JSON_profile_info


class CatalogInfo(object):
    def __init__(self, nrows: int, ncols: int, dataset_name: str, data_source_path, file_format: str, schema_info: dict,
                 profile_info: dict):
        self.profile_info = profile_info
        self.schema_info = schema_info
        self.file_format = file_format
        self.dataset_name = dataset_name
        self.data_source_path = data_source_path
        self.ncols = ncols
        self.nrows = nrows


def load_data_source_profile(data_source_path: str, file_format: str):
    profile_info = dict()
    schema_info = dict()
    ncols = 0
    nrows = 0
    dataset_name = None
    source_path = None
    for d in os.listdir(data_source_path):
        files = [f for f in os.listdir(f'{data_source_path}/{d}/')]
        for f in files:
            profile = load_JSON_profile_info(f'{data_source_path}/{d}/{f}')
            profile_info[profile.column_name] = profile
            schema_info[profile.column_name] = profile.data_type

            ncols += 1
            nrows = max(profile.total_values_count, nrows)
            if dataset_name is None:
                dataset_name = profile.dataset_name
                source_path = profile.path

    return CatalogInfo(nrows=nrows, ncols=ncols, file_format="csv", dataset_name=dataset_name,
                       schema_info=schema_info, profile_info=profile_info, data_source_path=source_path)
