import fasttext.util

fasttext.util.download_model('en', if_exists='ignore')
import fasttext
import fasttext.util
import os
from os.path import dirname


def download_model():
    module_path = dirname(__file__)
    fname = f"{module_path}/cc.en.50.bin"
    if not os.path.exists(fname):
        ft = fasttext.load_model('cc.en.300.bin')
        fasttext.util.reduce_model(ft, 50)
        ft.save_model(fname)
