from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
import copy
import warnings
import random, re
import itertools
import numpy as np
import pandas as pd
import copy
import math
import scipy
import time
import argparse
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings
from flair.data import Sentence

warnings.filterwarnings("ignore")
random.seed(100)


def DeduplicateData(data_down_stream, input_down_path, duplicate_column):
    df = input_down_path
    df = df.sort_values(by=['times_entered'], ascending=False)
    df = df.sort_values(by=['group'])
    df = df.reset_index()

    prvgrpno = 0
    dicdups = {}
    curdup = ''
    for index, row in df.iterrows():
        curgrpno = row['group']
        if prvgrpno != curgrpno:
            curdup = row[duplicate_column]
            dicdups[curdup] = curdup
        else: dicdups[row[duplicate_column]] = curdup
        prvgrpno = curgrpno

    dataDownstream_dedup = copy.deepcopy(data_down_stream)
    dataDownstream_dedup[duplicate_column] = dataDownstream_dedup[duplicate_column].fillna('0')
    def func(x):
        if x == '0': return '0'
        return dicdups[x]

    dataDownstream_dedup[duplicate_column] = dataDownstream_dedup[duplicate_column].apply(lambda x: func(x))
    attribute_names = dataDownstream_dedup.columns.values.tolist()

    dupcol_lst_values = dataDownstream_dedup[duplicate_column].values.tolist()
    embed_lst = []
    for word in dupcol_lst_values:
        sentence = Sentence(word)
        embedding.embed(sentence)
        tmp_tensor = sentence.embedding
        tmp_lst = tmp_tensor.tolist()
        embed_lst.append(tmp_lst)

    arr = np.array(embed_lst)
    df = pd.DataFrame(arr)
    dataDownstream_dedup = pd.concat([dataDownstream_dedup,df], axis=1, sort=False)

    return dataDownstream_dedup
    # return None
