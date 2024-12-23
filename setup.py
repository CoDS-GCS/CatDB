from setuptools import setup, find_packages


setup(
    name="dataprofiling",
    version="0.0.1",
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
        "bitstring>=3.1.9",
        "chars2vec>=0.1.7",
        "dateparser>=1.2.0",
        "fasttext>=0.9.3",
        "nltk>=3.9.1",
        "numpy==1.26.4",
        "pandas>=2.2.2",
        "pyspark==3.5.3",
        "spacy",
        "torch>=2.5.1",
        "tqdm>=4.67.1",
        "PyYAML==6.0.1",
        "keras==3.7.0",
        "tensorflow==2.18.0",
        "protobuf==5.29.2",
        #"py4j==0.10.9",
        "filesplit"
    ],
    package_data={
        "dataprofiling": ["column_embeddings/pretrained_models/date/*.pt", 
                          "column_embeddings/pretrained_models/general_string/*.pt",
                          "column_embeddings/pretrained_models/int/*.pt",
                          "column_embeddings/pretrained_models/natural_language/*.pt",
                          "column_embeddings/pretrained_models/numerical/*.pt",
                          "column_embeddings/pretrained_models/float/*.pt",
                          "column_embeddings/pretrained_models/natural_language_text/*.pt",
                          "column_embeddings/pretrained_models/string/*.pt",
                          "fasttext_embeddings/zip/*.zip",
                          "fasttext_embeddings/zip/manifest"]
    },
    zip_safe=False,
    # entry_points={
    #     "console_scripts": [
    #         "dataprofiling=dataprofiling.main:main",
    #     ]
    # }
)
