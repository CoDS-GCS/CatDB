
import mysql.connector

class DBDriver(object):
    def __init__(self, db_username = 'root', db_password = 'saeed@deeas', db_name='CatDB'):
        self.db_username = db_username
        self.db_password = db_password
        self.db_name = db_name

    def db_connection(self):
        cnx = mysql.connector.connect(
            host="localhost",
            user=self.db_username,
            password=self.db_password,
            database=self.db_name)
        return cnx
    def run_sql_script (self, script):
        cnx = self.db_connection()
        cursor = cnx.cursor()
        result = cursor.execute(script)
        rows = cursor.fetchall()
        cnx.close()
        return True, rows

    def run_sql_insert_commit (self, script):
        cnx = self.db_connection()
        cursor = cnx.cursor()
        cursor.execute(script)
        cnx.commit()
        cnx.close()
    def get_table_script (self, tbl_name):
        # drop table if exists CatDB.{tbl_name};
        tbl_script = (f'drop table if exists CatDB.{tbl_name}; \
                            create table if not exists CatDB.{tbl_name}( \
                            attribute_name        varchar(100) not null primary key,\
                            col_dtype             varchar(30)  not null, \
                            col_count             int          not null, \
                            col_unique            tinyint(1)   not null, \
                            col_top               varchar(256) null, \
                            col_mean              float        null, \
                            col_std               float        null, \
                            col_min               float        null, \
                            col_max               int          null, \
                            col_categorical       tinyint(1)   not null, \
                            col_categorical_data  text         null, \
                            col_categorical_count int          null, \
                            col_nullable          int          null, \
                            col_null_count        int          not null, \
                            constraint {tbl_name}_attribute_name_uindex unique (attribute_name)\
                        );')
        return tbl_script

if __name__ == '__main__':
    cnx = DBDriver(db_username='root', db_password='saeed@deeas', db_name='CatDB')
    mystr = cnx.get_table_script("TBLSaeed1")
    print(mystr)
    status, result =  cnx.run_sql_script(mystr)
    print(status)
