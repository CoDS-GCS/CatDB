
class GenerateReader(object):
    def __init__(self, file_format: str):
        self.src_package = []
        self.src_code =[]
        self.file_format = file_format
        self.generate_reader = None
        if file_format.lower() == "csv":
            self.generate_reader = self.generate_reader_csv
        else:
            raise Exception("The file format is not supported!")

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
        self.generate_reader(target_name="train_data", fname=train_fname, header=header, cols_dtypes=cols_dtypes)
        self.generate_reader(target_name="test_data", fname=test_fname, header=header, cols_dtypes=cols_dtypes)

    def load_dataset(self, fname: str, header=None, cols_dtypes: dict = None):
        self.generate_reader(target_name="data", fname=fname, header=header, cols_dtypes=cols_dtypes)

    def load_train_test_validation_dataset(self, train_fname: str, test_fname: str, validation_fname: str, header=None,
                                           cols_dtypes: dict = None):
        self.generate_reader(target_name="train_data", fname=train_fname, header=header, cols_dtypes=cols_dtypes)
        self.generate_reader(target_name="test_data", fname=test_fname, header=header, cols_dtypes=cols_dtypes)
        self.generate_reader(target_name="val_data", fname=validation_fname, header=header, cols_dtypes=cols_dtypes)
