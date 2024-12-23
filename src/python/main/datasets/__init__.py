import zipfile
import os
from os.path import dirname
from typing import List


def _extract_zip(dir_name):
    module_path = dirname(__file__)
    name = f"{module_path}/{dir_name}"
    with zipfile.ZipFile(f"{name}.zip", 'r') as zip_ref:
        if not os.path.exists(name):
            zip_ref.extractall(module_path)


def get_dataset_names() -> List[str]:
    module_path = dirname(__file__)
    _extract_zip(dir_name="data")
    files = os.listdir(f"{module_path}/data")
    datasets = list(files)

    return datasets


def get_catalogs() -> List[str]:
    module_path = dirname(__file__)
    _extract_zip(dir_name="catalog")
    files = os.listdir(f"{module_path}/catalog")
    catalogs = list(files)

    return catalogs


def get_dataset_path(name: str) -> str:
    if name not in get_dataset_names():
        raise ValueError(
            f"Dataset {name} is not found. You may want to try get_dataset_names()"
            + " to get all available dataset names"
        )

    module_path = dirname(__file__)
    path = f"{module_path}/data/{name}"
    return path


def get_catalog_path(name: str) -> str:
    if name not in get_catalogs():
        raise ValueError(
            f"Dataset {name} is not found. You may want to try get_catalogs()"
            + " to get all available data catalogs for datasets"
        )

    module_path = dirname(__file__)
    path = f"{module_path}/catalog/{name}"
    return path


def get_dataset_metadata_path(name: str) -> str:
    if name not in get_dataset_names():
        raise ValueError(
            f"Dataset {name} is not found. You may want to try get_dataset_names()"
            + " to get all available dataset names"
        )

    module_path = dirname(__file__)
    path = f"{module_path}/data/{name}/{name}.yaml"
    if os.path.exists(path):
        return path
    else:
        raise ValueError(f"Metadata for Dataset {name} is not found.")


def get_root_data_path() -> str:
    module_path = dirname(__file__)
    path = f"{module_path}/data/"
    return path


def get_root_catalog_path() -> str:
    module_path = dirname(__file__)
    path = f"{module_path}/catalog/"
    return path

__all__ = ["get_dataset_path",
           "get_dataset_names",
           "get_catalog_path",
           "get_catalogs",
           "get_dataset_metadata_path",
           "get_root_data_path"]
