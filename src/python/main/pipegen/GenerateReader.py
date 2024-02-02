class GenerateReader(object):
    def __init__(self, file_format: str):
        self.src_package = []
        self.src_code = []
        self.file_format = file_format
        self.generate_reader = None
        if file_format.lower() == "csv":
            self.generate_reader = self.generate_reader_csv
        else:
            raise Exception("The file format is not supported!")

    def get_reader(self, fname: str = None, train_fname: str = None, test_fname: str = None,
                   validation_fname: str = None,
                   header=None, cols_dtypes: dict = None):
        if fname is not None:
            self.load_dataset(fname=fname, header=header, cols_dtypes=cols_dtypes)
        elif train_fname is not None and test_fname is not None and validation_fname is not None:
            self.load_train_test_validation_dataset(train_fname=train_fname, test_fname=test_fname,
                                                    validation_fname=validation_fname, header=header,
                                                    cols_dtypes=cols_dtypes)
        elif train_fname is not None and test_fname is not None:
            self.load_train_test_dataset(train_fname=train_fname, test_fname=test_fname, header=header,
                                         cols_dtypes=cols_dtypes)
        else:
            raise Exception("The file name(s) is required for load dataset!")

        return self.src_package, self.src_code

    def generate_reader_csv(self, target_name: str, fname: str, header=None, cols_dtypes: dict = None):
        self.src_package.append("import pandas as pd")
        src = [f"{target_name} = pd.read_csv({fname}"]
        if header is not None:
            src.append(f", header = {header}")
        if cols_dtypes is not None:
            src.append(", dtype={")
            src.append(f"{cols_dtypes}")
            src.append("}")
        src.append(")")

        self.src_code.append("".join(src))

    def load_train_test_dataset(self, train_fname: str, test_fname: str, header=None, cols_dtypes: dict = None):
        self.generate_reader(target_name="df_train", fname=train_fname, header=header, cols_dtypes=cols_dtypes)
        self.generate_reader(target_name="df_test", fname=test_fname, header=header, cols_dtypes=cols_dtypes)

    def load_dataset(self, fname: str, header=None, cols_dtypes: dict = None):
        self.generate_reader(target_name="data", fname=fname, header=header, cols_dtypes=cols_dtypes)
        self.src_package.append("from sklearn.model_selection import train_test_split")
        self.src_code.append(" df_train, df_test = train_test_split(data, test_size=0.3, random_state=42)")

    def load_train_test_validation_dataset(self, train_fname: str, test_fname: str, validation_fname: str, header=None,
                                           cols_dtypes: dict = None):
        self.generate_reader(target_name="df_train", fname=train_fname, header=header, cols_dtypes=cols_dtypes)
        self.generate_reader(target_name="df_test", fname=test_fname, header=header, cols_dtypes=cols_dtypes)
        self.generate_reader(target_name="df_val", fname=validation_fname, header=header, cols_dtypes=cols_dtypes)
