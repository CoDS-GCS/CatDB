import os
from os.path import dirname

def download_model():
    import fasttext.util
    fasttext.util.download_model('en', if_exists='ignore')
    import fasttext
    import fasttext.util
    module_path = dirname(__file__)
    fname = f"{module_path}/cc.en.50.bin"
    if not os.path.exists(fname):
        ft = fasttext.load_model('cc.en.300.bin')
        fasttext.util.reduce_model(ft, 50)
        ft.save_model(fname)


def extract_model():
    from filesplit.merge import Merge
    import zipfile
    # from filesplit.split import Split
    # split = Split(inputfile="cc.en.50.bin.zip", outputdir="zip/")
    # split.bysize(size=94371840)

    module_path = dirname(__file__)
    fname = f"{module_path}/cc.en.50.bin"
    if not os.path.exists(fname):
        fname_zip = f"{module_path}/zip/"
        final_fname = f"{module_path}/cc.en.300.bin.zip"
        merge = Merge(inputdir=fname_zip, outputdir=module_path, outputfilename="cc.en.300.bin.zip")
        merge.merge()
        with zipfile.ZipFile(f"{final_fname}", 'r') as zip_ref:
            if not os.path.exists(fname):
                zip_ref.extractall(module_path)