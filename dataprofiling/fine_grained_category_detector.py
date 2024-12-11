import warnings

warnings.simplefilter('ignore')
import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # set tensorflow log level to FATAL

import spacy
import dateparser
import fasttext

fasttext.FastText.eprint = lambda *args, **kwargs: None
import pandas as pd
from nltk.tokenize import TweetTokenizer
import numpy as np

from model.column_data_type import ColumnDataType


class FineGrainedColumnCategoryDetector:
    try:
        ner_model = spacy.load('en_core_web_sm')
    except:
        import subprocess
        subprocess.call('python -m spacy download en_core_web_sm'.split(), shell=False)
        ner_model = spacy.load('en_core_web_sm')

    fasttext_model = fasttext.load_model(str(Path(__file__).parent) + '/fasttext_embeddings/cc.en.50.bin')
    tokenizer = TweetTokenizer()

    @staticmethod
    def detect_column_category_type(column: pd.Series, categorical_ratio):

        column_object = column.infer_objects()
        samples = column.sample(min(len(column), 100)).dropna().tolist()

        if column_object.name == 'category':
            return FineGrainedColumnCategoryDetector.__make_categorical_column(column, column.dropna().unique().tolist(), samples)

        else:
            unique_values = list(column.dropna().unique())
            if column.dtype.type == np.bool_:
                return FineGrainedColumnCategoryDetector.__make_categorical_column(column, unique_values, samples)

            elif column.dtype.type in [np.int64, np.uint64]:
                if set(column.unique()) == {0, 1}:
                    return FineGrainedColumnCategoryDetector.__make_categorical_column(column, unique_values, samples)

            elif column.dtype.type == np.float64:
                if set(column.unique()) == {0.0, 1.0}:
                    return FineGrainedColumnCategoryDetector.__make_categorical_column(column, unique_values, samples)

            nrows = len(column)
            if len(unique_values) <= 30 or nrows * categorical_ratio >= len(unique_values):
                return FineGrainedColumnCategoryDetector.__make_categorical_column(column, unique_values, samples)

            else:
                return {'samples': samples, 'category': None}

    @staticmethod
    def __is_date(column: pd.Series):
        num_date_values = 0
        for value in column.values:
            # the value is a date if it is short enough and is parsed by the dateparser
            if len(value) < 50 and dateparser.parse(value, locales=['en-CA'], languages=['en'],
                                                    settings={'STRICT_PARSING': True}):
                num_date_values += 1
                if num_date_values > 0.5 * len(column):
                    return True
        return False

    @staticmethod
    def __make_categorical_column(column, values, samples):
        category_values_ratio = column.value_counts(dropna=True).to_dict()
        result = {'category': ColumnDataType.CATEGORY,
                  'category_values': values,
                  'category_values_ratio': category_values_ratio,
                  'samples': samples}

        return result