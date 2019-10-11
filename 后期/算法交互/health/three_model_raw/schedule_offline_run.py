# coding=utf-8
import random
import time
import schedule
from utils import common_config as cf
from utils.db_connect import EsgynDB
from utils.file_load import FileLoad
from utils.model_util import Model_Util as model_util

"""
    从数据库取数据到模型，也就是训练
    500 * 2048 = 1024000 ---> record 1024
    500 * 159744 = 79872000‬ ---> record 79872
"""


class ScheduleJob:

    def __init__(self):
        common_config = cf.get_common_config()
        self.breakdown_type_config = cf.get_breakdown_type()
        self.breakdown_type = {0: 0}
        self.file_load = FileLoad()
        self.model = model_util()
        self.offline_config = cf.get_offline_config()
        # 标签的个数
        self.label_num = 3
        # 单个样本的个数
        self.spot_num = 2048
        # 样本数
        self.sample_num = 500
        self.count_times = common_config['db']['schedule']['count_iter_times']
        self.fetch_size = common_config['db']['schedule']['fetch_size']
        self.count_number = common_config['db']['schedule']['count_number']
        self.get_breakdown_type()
        self.by_count_job = schedule.every(self.count_times).seconds.do(self.schedule_by_count)  # minutes

    def get_breakdown_type(self):
        esgynDB = EsgynDB()
        sql = 'select BREAKDOWN_TYPE_ID,BREAKDOWN_TYPE from INTERPRETABLE_BREAKDOWN_TYPE;'
        iter_result = esgynDB.query(sql, 10)
        for rows in iter_result:
            for row in rows:
                if row[1] in self.breakdown_type_config:
                    self.breakdown_type[row[0]] = self.breakdown_type_config[row[1]]
                else:
                    self.breakdown_type[row[0]] = 0

    def execute(self, esgynDB, sensor_id, old_time):
        end_time = None
        if old_time:
            sql = "SELECT sample_value,flag,sample_ts FROM BREAKDOWN_DATA where sensor_id = '%s' and sample_ts > '%s' order by sample_ts" % (
            sensor_id, old_time)
        else:
            sql = "SELECT sample_value,flag,sample_ts FROM BREAKDOWN_DATA where sensor_id = '%s' order by sample_ts" % sensor_id
        iter_result = esgynDB.query(sql, self.fetch_size)
        end_row = None
        data = [[] for i in range(0, self.label_num)]
        for rows in iter_result:
            end_row = rows
            for row in rows:
                # label 0 ,1 ,2
                label = self.breakdown_type[row[1]]
                data[label] = data[label] + list(map(float, row[0].replace('[', '').replace(']', '').split(',')))
        else:
            if end_row is not None:
                end_time = end_row.pop()[2]
                # print('endTime: ', endTime, '<=====最后一次数据数====>', row.__len__())
                # print('datetime', end_time)  # datetime.datetime.fromtimestamp(float(format(endTime/1000.0, '.3f')))
                samples_train, labels_train = self.handle_data(data)
                self.call_model(sensor_id, samples_train, labels_train)
                # print('samples_train', len(samples_train))
                # print('labels_train', len(labels_train))
        return end_time

    def handle_data(self, data):
        samples_train = []
        labels_train = []
        length = len(data)
        for index in range(0, length):
            data1 = data[index]
            data1_len = len(data1)
            if data1_len < self.spot_num and data1_len != 0:
                value = data1[0]
                for i in range(data1_len, self.spot_num):
                    data1.append(value)

            for i in range(0, len(data1), self.spot_num):
                tmp = data1[i:i + self.spot_num]
                if len(tmp) == self.spot_num:
                    samples_train.append(tmp)
                    label = [0 for i in range(0, self.label_num)]
                    label[index] = 1
                    labels_train.append(label)
        samples_len = len(samples_train)
        for index in range(samples_len, self.sample_num):
            num = int(random.random() * samples_len)
            samples_train.append(samples_train[num])
            labels_train.append(labels_train[num])
        return samples_train, labels_train

    def call_model(self, sensor_id, samples_train, labels_train):
        # file_path '/a/b/checkpoints/*'  version 'v1'
        file_path, version = self.model.call_model_train(sensor_id, samples_train, labels_train)
        if file_path is not None and version is not None:
            self.file_load.upload_checkpoints(sensor_id, file_path, version)
            self.file_load.db.close()

    def schedule_by_count(self):
        esgynDB = EsgynDB()
        for sensor_id in self.offline_config:
            # 如果需要这么多点的就打开这个判断，否则默认2048个点
            if sensor_id == "cDAQ9189-1D91958Mod5/ai1":
                self.spot_num = 159744
                self.count_number = 79872
            old_time = self.offline_config[sensor_id]
            if old_time:
                sql = "SELECT count(1) FROM BREAKDOWN_DATA where sensor_id = '%s' and sample_ts > '%s'" % (sensor_id, old_time)
            else:
                sql = "SELECT count(1) FROM BREAKDOWN_DATA where sensor_id = '%s'" % sensor_id
            row_count = esgynDB.count(sql)[0]
            if row_count >= self.count_number:
                end_time = self.execute(esgynDB, sensor_id, old_time)
                if end_time:
                    self.offline_config[sensor_id] = str(end_time)
        esgynDB.close()
        # 更新时间
        cf.update_offline_config(self.offline_config)


if __name__ == '__main__':
    job = ScheduleJob()
    while True:
        schedule.run_pending()
        time.sleep(1)
