from catalog.Catalog import CatalogInfo


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
        self.target_attribute: target_attribute
        self.train_fname: train_fname
        self.test_fname: test_fname
        self.validation_fname: validation_fname

        self.src = []

    def get_pipeline(self):
        # 1. collect all packages are required
        # 2. generate dataset reader
        # 3. clean data
        # 4. apply feature engineering/ data transformation
        # 5. build model
        # 6. report results
        import_packages = []
