import pandas

from src.main.python.util import DatasetReader as reader
from src.main.python.catalog.Profile import ProfileInfo
from src.main.python.catalog.Profile import get_profile_info
from src.main.python.catalog.Profile import get_schema_info


class CatalogInfo(object):
    def __init__(self, nrows: int, ncols: int, dataset_name: str, file_format: str, schema_info: dict,
                 profile_info: ProfileInfo):
        self.profile_info = profile_info
        self.schema_info = schema_info
        self.file_format = file_format
        self.dataset_name = dataset_name
        self.ncols = ncols
        self.nrows = nrows


def get_data_catalog(dataset_name=None, file_format='csv'):
    data = read_dataset(dataset_name=dataset_name, file_format=file_format)
    schema_info = get_schema_info(data=data)
    profile_info = get_profile_info(data=data, schema_info=schema_info)
    if data is not None:
        nrows = data.shape[0]
        ncols = data.shape[1]
    else:
        nrows = -1
        ncols = -1
    return CatalogInfo(nrows=nrows, ncols=ncols, dataset_name=dataset_name, file_format=file_format,
                       schema_info=schema_info, profile_info=profile_info)


def read_dataset(dataset_name: str, file_format: str):
    dr = reader.DatasetReader(dataset_name)

    df = pandas.DataFrame
    if file_format == 'csv':
        df = dr.reader_CSV()

    elif file_format == 'json':
        df = dr.reader_JSON()
    return df
