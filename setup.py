from setuptools import setup, find_packages
import os
from os.path import dirname
import shutil

# module_path = dirname(__file__)

# src_dir = f'{module_path}/src/'
# dest_dir = f'{module_path}/src/dataprofiling'

# if os.path.exists(dest_dir) and os.path.isdir(dest_dir):
#     shutil.rmtree(dest_dir)

# # getting all the files in the source directory
# files = os.listdir(src_dir)
# shutil.copytree(src_dir, dest_dir)

setup(
    name="dataprofiling",
    version="0.0.2",
    packages=find_packages(),
    # package_dir={'': 'src'},
    # py_modules=['dataprofiling'],
    description="KGLiDS employs machine learning and knowledge graph technologies to abstract and capture the semantics of data science artifacts and their connections.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Saeed Fathollahzadeh, Essam Mansour",
    author_email="s.fathollahzadeh@gmail.com",
    url="https://github.com/CoDS-GCS/kglids",
    license="LICENSE.txt",
    classifiers=[
        "Development Status :: 4 - Alpha",
        "License :: Free for non-commercial use",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12"
    ],
    python_requires=">=3.10",    
    install_requires=[
        "bitstring==3.1.9",
        # "camelsplit==1.0.1",
        # "h5py==3.1.0",
        # "matplotlib==3.7.0",
        # "nltk==3.5",
        # "py==1.10.0",
        # "py-rouge==1.1",
        # "py4j==0.10.9",
        # "pyspark==3.0.1",
        # "pytest==6.2.2",
        # # "scikit-learn==1.0.2",
        # "scikit-learn",
        # "spacy==3.4.1",
        # "SPARQLWrapper==1.8.5",
        # "torch==1.12.0",
        # "tqdm==4.56.0",
        # "astor==0.8.1",
        # "graphviz==0.17",
        # "numpy==1.21.4",
        # # "pandas==1.3.4",
        # "pandas>=2.2.2",
        # "staticfg==0.9.5",
        # "urllib3==1.26.7",
        # "seaborn~=0.11.2",
        # # "dask~=2022.2.1",
        # "dask[complete]",
        # "python-dateutil==2.8.2",
        # "dateparser==1.1.1",
        # "chars2vec==0.1.7",
        # "keras==2.8.0",
        # "tensorflow==2.8.0",
        # "protobuf==3.19.1",
        # "psycopg==3.1.10",
        # "jupyter==1.0.0",
        # "pystardog==0.16.1",
        # "typing-inspect==0.8.0",
        # "typing_extensions==4.5.0",
        # "memory_profiler",
        # "ogb",
        # "torch_sparse",
        # "torch_geometric"
    ],
    # package_data={
    #     "catdb": ["*.yaml", "datasets/*.zip", "ui/templates/*.html"]
    # },
    zip_safe=False,
    # entry_points={
    #     "console_scripts": [
    #         "catdb=catdb.main:main",
    #     ]
    # }
)