# coding=utf-8
import time
import os

from utils.db_connect import EsgynDB

"""
    upload_checkpoints 上传文件，local_path要具体到相应的目录 checkpoints/
    down_checkpoints 下载文件，local_path要指定一个目录，会在这个目录下生成一个checkpoints目录，文件在checkpoints目录下
"""


class FileLoad:
    def __init__(self):
        self.hdfUtils = HdfUtils()
        self.db = EsgynDB()

    def upload_checkpoints(self, sensor_id, local_path, version):
        """
        :param version: 版本号
        :param local_path: 具体到相应文件的目录 checkpoints/
        :return:
        """
        date = time.time()
        child_path = int(date) 
        hdfs_path = "/machine_learning/%s/" % child_path
        self.hdfUtils.put_to_hdfs(local_path, hdfs_path)
        # 把路径保存到数据库中
        date = int(date  * 1000)
        self.db.insert("insert into ml_file_path (sensor_id, path, version, created) values ('%s', '%s', '%s', %s);" % (sensor_id, hdfs_path, version, date))

    def down_checpoints(self, sensor_id, date, local_path):
        """
        :param date: %Y-%m-%d %H:%M:%S eg: 2019-08-01 14:21:23
        :param local_path:
        :return:
        """
        local_path = local_path + sensor_id.replace('/','_') + "/"
        if not os.path.isdir(local_path):
            os.mkdir(local_path)
        date = int(time.mktime(time.strptime(date, "%Y-%m-%d %H:%M:%S")) * 1000)
        sql = "select * from ml_file_path where created >= '%s' and sensor_id = '%s'" % (date, sensor_id)
        # 查找数据库获得所需文件的路径
        result = self.db.getMLPath(sql)

        for record in result:
            #  具体到日期就好
            hdfs_path = record[1]
            self.hdfUtils.get_from_hdfs(hdfs_path, local_path)


class HdfUtils:

    # 上传文件到hdfs
    def put_to_hdfs(self, local_path, hdfs_path):
        cmd_mkdir = "su hdfs -c 'hdfs dfs -mkdir %s'" % hdfs_path
        cmd_chown = "su hdfs -c 'hdfs dfs -chown root:root %s'" % hdfs_path
        cmd_put = 'hdfs dfs -copyFromLocal %s %s' % (local_path, hdfs_path)
        print(cmd_put)
        os.system(cmd_mkdir)
        os.system(cmd_chown)
        result = os.system(cmd_put)
        print(result)

    # 从hdfs获取文件到本地
    def get_from_hdfs(self, hdfs_path, local_path):
        cmd = 'hdfs dfs -copyToLocal %s %s' % (hdfs_path, local_path)
        print(cmd)
        result = os.system(cmd)
        print(result)

    # 删除hdfs文件
    def delete_hdfs_file(self, hdfs_path):
        cmd = "su hdfs -c 'hdfs dfs -rm -r -f %s'" % hdfs_path
        print(cmd)
        result = os.system(cmd)
        print(result)



if __name__ == '__main__':
    hdfUtils = HdfUtils()
    #hdfUtils.get_from_hdfs('/machine_learning', 'b/')
    #hdfUtils.delete_hdfs_file('/machine_learning/logs')
    #hdfUtils.put_to_hdfs('MT2_model_checkpoints/axis_checkpoints/*', '/machine_learning/a/')
    #机器学习产生的结果保存起来
    file_load = FileLoad()
    #file_load.upload_checkpoints("cDAQ2Mod3/ai0", "/wzs/model/three_model/MT2_model_checkpoints/axis_checkpoints/*", 'v2')
    file_load.down_checpoints("cDAQ2Mod3/ai0", "2019-08-14 00:00:00",  "a/")
    file_load.db.close()

