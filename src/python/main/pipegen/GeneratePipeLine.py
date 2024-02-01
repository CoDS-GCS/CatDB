from catalog.Catalog import CatalogInfo
from .GenerateReader import GenerateReader


class GeneratePipeLine(object):
    def __init__(self, catalog: CatalogInfo,
                 target_attribute: str,
                 train_fname: str,
                 test_fname: str,
                 validation_fname: str):
        self.schema_info = catalog.schema_info
        self.profile_info = catalog.profile_info
        self.nrows = catalog.nrows
        self.file_format = catalog.file_format
        self.target_attribute = target_attribute
        self.train_fname = train_fname
        self.test_fname = test_fname
        self.validation_fname = validation_fname

        self.src = []

    def get_pipeline(self):
        # 1. collect all packages are required
        # 2. generate dataset reader
        # 3. clean data
        # 4. apply feature engineering/ data transformation
        # 5. build model
        # 6. report results
        src_package = set()
        src_code = []

        # Convert data profiling data types to Python data types:
        cols_dtypes = dict()
        for col in self.schema_info.keys():
            dtype = self.schema_info[col]
            if dtype == "boolean":
                dtype = bool
            elif dtype not in {"int", "float", "date"}:
                dtype = "str"
            cols_dtypes[col] = dtype
        reader_src_package, reader_src_code = GenerateReader(file_format=self.file_format).get_reader(
            train_fname=self.train_fname,
            test_fname=self.test_fname, header=True,
            cols_dtypes=cols_dtypes)
        src_package.add(reader_src_package)
        src_code.append(reader_src_code)

        src_pipeline = "\n".join([
            "#``` python import packages",
            src_package,
            "#``` end python import packages\n",
            "# ``` python load dataset",
            src_code,
            "#``` python end load dataset \n"
        ])
        return src_pipeline
