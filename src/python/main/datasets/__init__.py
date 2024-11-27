import zipfile
import os
from os.path import dirname
from typing import List



# catalog_path = f'{module_path}/catalog'
# catalog_zip_path = f'{module_path}/catalog.zip'


def _extract_zip(data, zip):
    with zipfile.ZipFile(zip) as zip_file:
        for member in zip_file.namelist():
            if not os.path.exists(data):
                zip_file.extract(member, data)


def get_dataset_names() -> List[str]:
    module_path = dirname(__file__)
    data_path = f'{module_path}/data'
    data_zip_path = f'{module_path}/data.zip'

    _extract_zip(data=data_path, zip=data_zip_path)
    files = os.listdir(f"{module_path}/data")
    datasets = list(files)

    return datasets


def get_dataset_path(name: str) -> str:
    if name not in get_dataset_names():
        raise ValueError(
            f"Dataset {name} is not found. You may want to try get_dataset_names()"
            + " to get all available dataset names"
        )

    module_path = dirname(__file__)
    path = f"{module_path}/data/{name}"
    return path


__all__ = ["get_dataset_path", "get_dataset_names"]