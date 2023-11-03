import DatasetReader as reader
import DBDriver as db
import numpy as np
import sys


class Catalog(object):
    def __init__(self, dsname = None, dsroot = None, fileformat = None):
        self.dsname = dsname
        self.fileformat = fileformat
        self.db = db.DBDriver()
        self.nrow = 0
        self.categorical_ratio = 300
        self.tbl_name = None
        self.dsroot = dsroot

    def read_dataset(self):
        dr = reader.DatasetReader(self.dsname)
        self.df = None

        if self.fileformat == 'csv':
            self.df = dr.reader_CSV()
        elif self.fileformat == 'json':
            self.df = dr.reader_JSON()

        if self.df is not None:
            self.nrow = len(self.df)

    def create_ds_schema(self):
        ds_path = self.dsname.split('/')
        tbl_name = ds_path[len(ds_path) - 1]
        tbl_name = tbl_name[:len(tbl_name) - len(self.fileformat) - 1]
        tbl_name = tbl_name.replace('.', '_')
        if self.dsroot !=None and self.dsroot!='':
            self.tbl_name = f'{self.dsroot}_{tbl_name}'
        else:
            self.tbl_name = tbl_name

        tbl_script = self.db.get_table_script(tbl_name=self.tbl_name)
        status, result = self.db.run_sql_script(tbl_script)

        return status

    def compute_statistic(self, colname, datatype):
        if isinstance(datatype, str) or isinstance(datatype, bool) or isinstance(datatype, object):
            return 'null', 'null', 'null', 'null'
        else:
            col_min = np.min(self.df[colname])
            col_max = np.max(self.df[colname])
            col_mean = np.mean(self.df[colname])
            col_std = np.std(self.df[colname])
            return col_min, col_max, col_mean, col_std

    def compute_counts(self, colname):
        null_count = self.df[colname].isna().sum()
        nullable = null_count > 0
        count = self.nrow - null_count
        return count, nullable, null_count

    def compute_groups(self, colname):
        unique_values = self.df[colname].unique()
        unique_len = len(unique_values)
        is_unique = unique_len == self.nrow
        is_categorical = unique_len <= self.categorical_ratio
        if is_categorical:
            return is_unique, is_categorical, unique_values, unique_len
        else:
            return is_unique, is_categorical, '', '0'

    def add_catalog_data(self):
        self.read_dataset()
        self.create_ds_schema()
        ds_cols = self.df.columns

        sql_insert = f'INSERT INTO {self.tbl_name}(attribute_name, col_dtype, col_count, col_unique, col_top, col_mean, col_std, col_min, col_max, col_categorical, col_categorical_data, col_categorical_count, col_nullable, col_null_count) VALUES'
        for col_name in ds_cols:

            attribute_name = col_name
            col_dtype = self.df.dtypes[col_name]

            col_count, col_nullable, col_null_count = self.compute_counts(col_name)
            col_unique, col_categorical, col_categorical_data, col_categorical_count = self.compute_groups(col_name)
            col_min, col_max, col_mean, col_std = self.compute_statistic(col_name, col_dtype)

            col_categorical_value = 'null'
            if col_categorical:
                col_categorical_value = f"{','.join([str(item) for item in col_categorical_data])}"
                col_categorical_value = col_categorical_value.replace('"',"")
                col_categorical_value = col_categorical_value.replace("'", "")
                col_categorical_value = f"'{col_categorical_value}'"

            sql_record = f"{sql_insert}('{attribute_name}','{col_dtype}',{col_count},{col_unique},null,{col_mean},{col_std},{col_min},{col_max},{col_categorical},{col_categorical_value},{col_categorical_count},{col_nullable},{col_null_count});"
            self.db.run_sql_insert_commit(sql_record)



if __name__ == '__main__':
    data_path = sys.argv[1]
    dsroot = sys.argv[2]
    dsnames = sys.argv[3].split(',')
    fileformat = sys.argv[4]

    for dn in dsnames:
        if len(dsroot) >= 1:
            dp = f'{data_path}/{dsroot}/{dn}'
            cat = Catalog(dsname=dp, dsroot= dsroot ,fileformat=fileformat)
        else:
            dp = f'{data_path}/{dn}'
            cat = Catalog(dsname=dp, fileformat=fileformat)

        cat.add_catalog_data()
