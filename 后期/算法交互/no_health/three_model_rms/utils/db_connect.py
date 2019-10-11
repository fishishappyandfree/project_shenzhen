# coding=utf-8
from pypyodbc import *
from utils.common_config import get_common_config


class EsgynDB:
    """
        EsgynDB connect by ODBC
    """
    def __init__(self):
        common_config = get_common_config()
        self.__connection = Connection(DSN=common_config['db']['dsn'], UID=common_config['db']['uid'], PWD=common_config['db']['pwd'])
        self.__cursor = self.__connection.cursor()
        self.__cursor.execute("SET SCHEMA FOXCONN")

    def count(self, sql):
        self.__cursor.execute(sql)
        return self.__cursor.fetchone()

    def insert(self, sql):
        self.__cursor.execute(sql)
        self.__cursor.commit()

    def getMLPath(self, sql):
        self.__cursor.execute(sql)
        rows = self.__cursor.fetchall()
        return rows

    def query(self, sql, fetch_size):
        self.__cursor.execute(sql)
        rows = self.__cursor.fetchmany(fetch_size)
        while rows:
            yield rows
            rows = self.__cursor.fetchmany(fetch_size)
    def query_all(self, sql):
        self.__cursor.execute(sql)
        return self.__cursor.fetchAll()

    def close(self):
        if self.__cursor:
            self.__cursor.close()
        if self.__connection:
            self.__connection.close()


if __name__ == '__main__':
    print(1)
