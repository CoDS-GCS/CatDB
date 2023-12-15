from src.main.python.util import DatasetReader as reader
from src.main.python.catalog import Profile

class Catalog(object):
    def __init__(self, file_path=None, dataset_name=None, file_format='csv'):
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.file_format = file_format
        self.df = None

        # catalog info
        self.nrows = -1
        self.ncols = -1
        self.col_names = None

    def read_dataset(self):
        dr = reader.DatasetReader(self.dataset_name)

        if self.file_format == 'csv':
            self.df = dr.reader_CSV()

        elif self.file_format == 'json':
            self.df = dr.reader_JSON()

        if self.df is not None:
            self.nrows = self.df.shape[0]
            self.ncols = self.df.shape[1]
            self.col_names = self.df.columns

    def profile_dataset(self):

        # read dataset
        if self.df is None:
            self.read_dataset()

        # get dataset info
        prof = Profile.Profile(data= self.df, nrows=self.nrows, columns=self.col_names)


        # df_describe = self.df.describe(include='all').T
        #
        # # organize dataset info into catalog info
        # self.nrows = self.df.shape[0]
        # self.ncols = self.df.shape[1]
        # self.col_names = self.df.columns
        #
        #
        # print(df_describe.columns)


