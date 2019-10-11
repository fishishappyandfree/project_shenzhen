# coding=utf-8
from __future__ import print_function
import copy
import time

from pypyodbc import *

# # 简单测试
# connection = Connection("DSN=traf;UID=trafodion;PWD=traf123")
# cursor = connection.cursor()
# cursor.execute("set schema foxconn")
# cursor.execute("select * from sensor")
# rows = cursor.fetchmany(10)
# while rows:
#     print(rows)
#     rows = cursor.fetchmany(10)


class EsgynDB:
    """
        通过 odbc 连接数据库批量消费数据
    """
    DEFAULT_CONFIG = {
        'DSN': 'traf',
        'UID': 'trafodion',
        'PWD': 'traf123',
        'fetch_size': 100
    }

    def __init__(self, **configs):
        self.config = copy.copy(self.DEFAULT_CONFIG)
        for key in self.config:
            if key in configs:
                self.config[key] = configs.pop(key)

        self.__fetch_size = self.config['fetch_size']
        self.__connection = Connection(DSN=self.config['DSN'], UID=self.config['UID'], PWD=self.config['PWD'])
        self.__cursor = self.__connection.cursor()

    def query(self, sql):
        self.__cursor.execute("SET SCHEMA FOXCONN")
        # self.__cursor.prepare("select count(1) from sensor")
        self.__cursor.execute(sql)
        rows = self.__cursor.fetchmany(self.__fetch_size)
        while rows:
            yield rows
            rows = self.__cursor.fetchmany(self.__fetch_size)

    def close(self):
        if self.__cursor:
            self.__cursor.close()
        if self.__connection:
            self.__connection.close()


if __name__ == '__main__':
    start = time.time()
    # fetch_size 代表一次拉取的数据条数
    esgynDB = EsgynDB(fetch_size=10)
    i = 1
    for rows in esgynDB.query('SELECT * FROM SENSOR'):  ##返回的是一个rows数组的可迭代对象generator
        print(i, rows)
        i = i + 1
    esgynDB.close()
    end = time.time()
    print(end - start)
