import pandas as pd

class DatasetReader(object):
    def __init__(self, filename):
        self.filename = filename
        self.data = None

    def reader_CSV (self):
        df = pd.read_csv(self.filename,  low_memory=False)
        return df

    def reader_JSON(self):
        df = pd.read_json(self.filename, lines=True)
        return df

