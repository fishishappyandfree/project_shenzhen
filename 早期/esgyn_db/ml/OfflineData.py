# coding=utf-8
from __future__ import print_function
import copy
import time
import schedule

from pypyodbc import *


class EsgynDB:
    """
        EsgynDB connect by ODBC
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

        self.__connection = Connection(DSN=self.config['DSN'], UID=self.config['UID'], PWD=self.config['PWD'])
        self.__cursor = self.__connection.cursor()

    def count(self, sql):
        self.__cursor.execute("SET SCHEMA FOXCONN")
        # self.__cursor.prepare("select count(1) from sensor")
        self.__cursor.execute(sql)
        return self.__cursor.fetchone()

    def query(self, sql, fetch_size):
        self.__cursor.execute("SET SCHEMA FOXCONN")
        # self.__cursor.prepare("select count(1) from sensor")
        self.__cursor.execute(sql)
        rows = self.__cursor.fetchmany(fetch_size)
        while rows:
            yield rows
            rows = self.__cursor.fetchmany(fetch_size)

    def close(self):
        if self.__cursor:
            self.__cursor.close()
        if self.__connection:
            self.__connection.close()


class ScheduleJob:

    __NUMBER = 100

    def __init__(self):
        self.startTime = None
        self.by_count_job = schedule.every(3).seconds.do(self.scheduleByCount)
        self.by_time_job = schedule.every(10).seconds.do(self.scheduleByTime)

    def execute(self):
        if self.startTime is None:
            sql = 'SELECT * FROM BREAKDOWN_DATA order by sample_ts'
        else:
            sql = "SELECT * FROM BREAKDOWN_DATA where sample_ts > '%s' order by sample_ts" % self.startTime
        esgynDB = EsgynDB()
        i = 1
        rows = esgynDB.query(sql, 10)
        endRow = None
        for row in rows:
            print(i, '取到了数据数: ', row.__len__())
            i = i + 1
            endRow = row
        else:
            if endRow is not None:
                endTime = row.pop()[2]
                # print('endTime: ', endTime, '<=====最后一次数据数====>', row.__len__())
                # print('datetime', endTime)  # datetime.datetime.fromtimestamp(float(format(endTime/1000.0, '.3f')))
                self.startTime = endTime
        esgynDB.close()

    def scheduleByTime(self):
        print("by time  ------>", datetime.datetime.now())
        self.execute()

    def scheduleByCount(self):
        esgynDB = EsgynDB()
        if self.startTime is None:
            sql = 'SELECT count(1) FROM BREAKDOWN_DATA'
        else:
            sql = "SELECT count(1) FROM BREAKDOWN_DATA where sample_ts > '%s'" % self.startTime

        rowCount = esgynDB.count(sql)
        esgynDB.close()
        print("by count ------>", datetime.datetime.now())
        if rowCount[0] != 0 and rowCount[0] >= self.__NUMBER:
            self.execute()
            schedule.cancel_job(self.by_time_job)
            self.by_time_job = schedule.every(10).seconds.do(self.scheduleByTime)


if __name__ == '__main__':
    job = ScheduleJob()
    while True:
        schedule.run_pending()
        time.sleep(1)
