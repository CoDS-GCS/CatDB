

from importlib.machinery import SourceFileLoader
load_data_source_profile = SourceFileLoader("load_data_source_profile", "/home/saeed/Documents/Github/CatDB/src/python/main/catalog/Catalog.py").load_module()


# from catalog.Catalog import load_data_source_profile
# from catalog.Dependency import load_dependency_info
# from pipegen.GeneratePipeLine import generate_and_verify_pipeline, run_pipeline, clean_categorical_data
# from util.FileHandler import read_text_file_line_by_line
# from util.Config import load_config
# from pipegen.Metadata import Metadata
# import time
# import yaml
# import os