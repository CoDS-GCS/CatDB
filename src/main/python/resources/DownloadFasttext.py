import fasttext.util

if __name__ == '__main__':
    fasttext.util.download_model('en', if_exists='ignore')
    ft = fasttext.load_model('cc.en.300.bin')
    fasttext.util.reduce_model(ft, 50)
    ft.save_model('cc.en.50.bin')