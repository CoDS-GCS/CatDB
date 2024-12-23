from datetime import datetime
import glob
import os
from pathlib import Path
import shutil
import warnings

warnings.simplefilter('ignore')

import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

import sys
from os.path import dirname

PACKAGE_PATH = dirname(__file__)
if PACKAGE_PATH not in sys.path:
    sys.path.append(PACKAGE_PATH)

from fine_grained_type_detector import FineGrainedColumnTypeDetector
from fine_grained_category_detector import FineGrainedColumnCategoryDetector
from profile_creators.profile_creator import ProfileCreator
from model.table import Table
from config import profiler_config, DataSource
from model.column_data_type import ColumnDataType
from raw_catalog.Catalog import load_data_source_profile


def _load_dataset_catalog(data_profile_path, root_data_path, root_catalog_path, dataset_name, metadata_path) -> dict:
    catalog = dict()
    catalog["data"] = [load_data_source_profile(data_source_path=data_profile_path)]
    catalog["root_data_path"] = root_data_path
    catalog["root_catalog_path"] = root_catalog_path
    catalog["data_profile_path"] = data_profile_path
    catalog["metadata_path"] = metadata_path
    catalog["dataset_name"] = dataset_name
    catalog["result_format"] = "catalog"
    return catalog


def build_catalog(data, categorical_ratio: float = 0.05, n_workers: int = -1, max_memory: float = -1):

    name = data["dataset_name"]
    path = data["root_data_path"]
    metadata_path = data["metadata_path"]
    output_path = f'{data["root_catalog_path"]}/data_profile'

    if name and path:
        extra_source = DataSource(name=name, path=path)
        profiler_config.data_sources.append(extra_source)
    if output_path:
        profiler_config.output_path = output_path
    if max_memory != -1:
        profiler_config.max_memory = max_memory
    else:
        profiler_config.max_memory = 1
    if n_workers != -1:
        profiler_config.n_workers = n_workers
    else:
        profiler_config.n_workers = os.cpu_count()
    start_time = datetime.now()
    print(datetime.now(), ': Initializing Spark')

    spark = None
    results = None
    try:
        # initialize spark
        if profiler_config.is_spark_local_mode:
            spark = SparkContext(conf=SparkConf().setMaster(f'local[{profiler_config.n_workers}]')
                                 .set('spark.driver.memory', f'{profiler_config.max_memory}g'))
        else:
            spark = SparkSession.builder.appName("CatDBDataProfiler").getOrCreate().sparkContext
            # add python dependencies
            for pyfile in glob.glob('./**/*.py', recursive=True):
                spark.addPyFile(pyfile)
            # add embedding model files
            for embedding_file in glob.glob('./column_embeddings/pretrained_models/**/*.pt', recursive=True):
                spark.addFile(embedding_file)
            # add fasttext embeddings file
            spark.addFile('./fasttext_embeddings/cc.en.50.bin')

        if os.path.exists(profiler_config.output_path):
            #print(datetime.now(), ': Deleting existing column profiles in:', profiler_config.output_path)
            shutil.rmtree(profiler_config.output_path)

        os.makedirs(profiler_config.output_path, exist_ok=True)

        # get the list of columns and their associated tables
        print(datetime.now(), ': Creating tables, Getting columns')
        columns_and_tables = []
        for data_source in profiler_config.data_sources:
            filenames = []
            if os.path.isfile(f"{path}/{name}.{data_source.file_type}"):
                filenames.append(f"{data_source.path}/{data_source.name}.{data_source.file_type}")
            else:
                for filename in glob.glob(os.path.join(data_source.path, '**/*.' + data_source.file_type), recursive=True):
                    if (filename.endswith(f"_train.{data_source.file_type}") or
                            filename.endswith(f"_test.{data_source.file_type}") or
                            filename.endswith(f"_verify.{data_source.file_type}")):
                        continue
                    else:
                        filenames.append(filename)

            for filename in filenames:
                if os.path.isfile(filename) and os.path.getsize(filename) > 0:  # if not an empty file
                    dataset_base_dir = Path(filename).resolve()
                    while dataset_base_dir.parent != Path(data_source.path).resolve():
                        dataset_base_dir = dataset_base_dir.parent
                    table = Table(data_source=data_source.name,
                                  table_path=filename,
                                  dataset_name=dataset_base_dir.name)
                    # read only the header
                    if len(filenames) > 1:
                        out_path = f"{output_path}/{table.table_name.replace(f'.{data_source.file_type}', '')}"
                    else:
                        out_path = output_path
                    header = pd.read_csv(table.get_table_path(), nrows=0, engine='python', encoding_errors='replace')
                    columns_and_tables.extend([(col, table, categorical_ratio, out_path) for col in header.columns])

        columns_and_tables_rdd = spark.parallelize(columns_and_tables)

        # profile the columns with Spark.
        print(datetime.now(), f': Profiling {len(columns_and_tables)} columns')
        columns_and_tables_rdd.map(column_worker).collect()

        print(datetime.now(), f': {len(columns_and_tables)} columns profiled and saved to {profiler_config.output_path}')
        print(datetime.now(), ': Total time to profile: ', datetime.now() - start_time)
        spark.stop()

        results = _load_dataset_catalog(data_profile_path= profiler_config.output_path, root_catalog_path=output_path,
                                     root_data_path=path, dataset_name= name, metadata_path=metadata_path)
    finally:
        if spark is not None:
            spark.stop()
    return results
def column_worker(column_name_and_table):
    column_name, table, categorical_ratio, out_path = column_name_and_table
    # read the column from the table file. Use the Python engine if there are issues reading the file
    try:
        try:
            column = pd.read_csv(table.get_table_path(), usecols=[column_name], na_values=[' ', '?', '-']).squeeze(
                "columns")
        except:
            column = pd.read_csv(table.get_table_path(), usecols=[column_name], na_values=[' ', '?', '-'],
                                 engine='python', encoding_errors='replace').squeeze("columns")
    except:
        print(f'Warning: Skipping non-parse-able column: {column_name} in table: {table.get_table_path()}')
        return

    nrows = len(column)
    column = pd.to_numeric(column, errors='ignore')
    column = column.convert_dtypes()
    column = column.astype(str) if column.dtype == object else column

    # infer the column data type
    column_type = FineGrainedColumnTypeDetector.detect_column_data_type(column)
    column_category = FineGrainedColumnCategoryDetector.detect_column_category_type(column, categorical_ratio)

    # collect statistics, generate embeddings, and create the column profiles
    column_profile_creator = ProfileCreator.get_profile_creator(column, column_type, table)
    column_profile = column_profile_creator.create_profile()

    if column_category is not None and column_category['category'] == ColumnDataType.CATEGORY:
        column_profile.set_category_values(column_category['category_values'])
        column_profile.set_category_values_ratio(column_category['category_values_ratio'])

    column_profile.set_samples(column_category['samples'])
    column_profile.set_nrows(nrows)
    # store the profile
    # column_profile.save_profile(profiler_config.output_path)
    column_profile.save_profile(out_path)


__all__ = ["build_catalog"]
