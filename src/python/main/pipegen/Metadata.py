from catalog.Catalog import CatalogInfo
from util.Config import REP_TYPE, PROFILE_TYPE_VAL


class Metadata(object):
    def __init__(self, catalog: CatalogInfo):
        self.catalog = catalog
        self.nrows = 5
        self.ncols = 11
        self.config_matrix = [[0 for column in range(self.ncols)] for row in range(self.nrows)]
        self.config_matrix[0] = [1 for column in range(self.ncols)]  # Tick all configs include Schema
        self.set_missing_values()
        self.set_numerical_values()
        self.set_distinct_values()
        self.set_categorical_values()

    def get_combinations(self):
        combinations = set()

        for col in range(self.ncols):
            s = 0
            for row in range(self.nrows):
                s += self.config_matrix[row][col] * PROFILE_TYPE_VAL[row]
            if REP_TYPE.get(s) is not None:
                combinations.add(REP_TYPE[s])
        return combinations

    def set_config(self, row, cols):
        for col in cols:
            self.config_matrix[row][col - 1] = 1

    def set_distinct_values(self):
        flag = False
        nrows = self.catalog.nrows
        for k in self.catalog.profile_info.keys():
            pi = self.catalog.profile_info[k]
            if pi.distinct_values_count != nrows:
                flag = True
                break
        if flag:
            cols = [2, 6, 7, 11]
            self.set_config(row=1, cols=cols)

    def set_missing_values(self):
        if len(self.catalog.columns_numerical_missing_values) > 0 or \
                len(self.catalog.columns_others_missing_values) > 0 or \
                len(self.catalog.columns_categorical_missing_values) > 0:
            cols = [3, 6, 8, 9, 11]
            self.set_config(row=2, cols=cols)

    def set_numerical_values(self):
        if len(self.catalog.columns_numerical):
            cols = [4, 7, 8, 1, 11]
            self.set_config(row=3, cols=cols)

    def set_categorical_values(self):
        if len(self.catalog.columns_categorical) > 0:
            cols = [5, 9, 10, 11]
            self.set_config(row=4, cols=cols)
