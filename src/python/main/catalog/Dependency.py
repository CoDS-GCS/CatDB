import yaml
import os


class Dependency(object):
    def __init__(self,
                 table_name: str = None,
                 columns: [] = None,
                 primary_keys = None,
                 foreign_keys = None,
                 ):
        self.table_name = table_name
        self.primary_keys = primary_keys
        self.foreign_keys = foreign_keys
        self.columns = columns

    def set_primary_keys(self, primary_keys):
        self.primary_keys = primary_keys

    def set_foreign_keys(self, foreign_keys):
        self.foreign_keys = foreign_keys

    def set_table_name(self, table_name):
        self.table_name = table_name

    def set_columns(self, columns):
        self.columns = columns


def load_dependency_info(dependency_file: str, datasource_name: str):
    if not os.path.isfile(dependency_file):
        return None

    with open(dependency_file, "r") as f:
        try:
            dep = yaml.load(f, Loader=yaml.FullLoader)
            ds_name = dep[0].get('name')
            if ds_name == datasource_name:
                tbls = dict()
                for k, v in dep[0].get('tables').items():
                    cols = dep[0].get('tables').get(k).get('columns').split(",")
                    PKs = dep[0].get('tables').get(k).get('PK')
                    FKs = dep[0].get('tables').get(k).get('FK')
                    d = Dependency(table_name=k, columns=cols)
                    if PKs is not None:
                        d.set_primary_keys(PKs.split(","))

                    if FKs is not None:
                        d.set_foreign_keys(FKs.split(","))

                    tbls[k] = d
                return tbls

        except yaml.YAMLError as ex:
            raise Exception(ex)
        except:
            return None
