from setuptools import setup, find_packages
import os
from os.path import dirname
import shutil

module_path = dirname(__file__)
src_dir = f'{module_path}/src/python/main'
dest_dir = f'{module_path}/src/catdb'

if os.path.exists(dest_dir) and os.path.isdir(dest_dir):
    shutil.rmtree(dest_dir)

# getting all the files in the source directory
files = os.listdir(src_dir)
shutil.copytree(src_dir, dest_dir)

setup(
    name="catdb",
    version="0.0.1",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=['catdb'],
    description="CatDB a comprehensive, LLM-guided generator of data-centric ML pipelines that utilizes available data catalog information. We incorporate data profiling information and user descriptions into a chain of LLM prompts for data cleaning/augmentation, feature engineering, and model selection. Additionally, we devise a robust framework for managing the LLM interactions and handling errors through pipeline modifications and a knowledge base of error scenarios.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Saeed Fathollahzadeh, Essam Mansour, Matthias Boehm",
    author_email="s.fathollahzadeh@gmail.com",
    url="https://github.com/CoDS-GCS/CatDB",
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
        "groq==0.5.0",
        "openai==1.6.1",
        "pandas>=2.2.2",
        "PyYAML==6.0.1",
        "tiktoken==0.7",
        "scikit-learn",
        "xgboost",
        "google-api-python-client",
        "google-auth-httplib2",
        "google-auth-oauthlib",
        "google-generativeai",
        "imbalanced-learn",
        "dask[complete]",
        "feature_engine",
        "unidecode",
        "flair",
        "torchvision"
    ],
    package_data={
        "catdb": ["*.yaml"],
    },
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "catdb=catdb.main:main",
        ]
    }
)