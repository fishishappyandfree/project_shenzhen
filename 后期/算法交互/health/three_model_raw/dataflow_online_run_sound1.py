# coding=utf-8
from utils.kafka_service import KafkaProducerService, KafkaConsumerService
from utils.model_util import Model_Util as model_util

"""
    从kafka消费数据到模型，也就是测试
"""


class DataInit(object):
    def __init__(self):
        self.model = model_util()
        self.producer = KafkaProducerService()
        self.consumer = KafkaConsumerService()
        self.result_dir = {}
        self.data_dir = {}

    def start_run(self):
        # 可以配置多个kafka topic
        for message in self.consumer.process('sensor', *['sound1']):
            sensor_id = message['sensor_id']
            #print(sensor_id)
            if not sensor_id in self.result_dir.keys():
                self.result_dir[sensor_id] = []
            if not sensor_id in self.data_dir.keys():
                self.data_dir[sensor_id] = []

            # 如果需要判断，把这一行注释掉，把下面的判断逻辑打开
            # self.call_model(sensor_id, message, 2048)
            if sensor_id == "cDAQ9189-1D91958Mod5/ai1":
                self.call_model(sensor_id, message, 159744)
            else:
                self.call_model(sensor_id, message, 2048)

    # 判断当前是否够 spot_num 个点，然后调用模型
    def call_model(self, sensor_id, message, spot_num):
        if len(self.data_dir[sensor_id]) >= spot_num:
            data = self.data_dir[sensor_id][:spot_num]
            self.data_dir[sensor_id] = []
            created = message['sample_ts']
            breakdownData = self.model.call_model_test(sensor_id, data, created)
            if breakdownData is not None:
                self.result_dir[sensor_id].append(1)
                if len(self.result_dir[sensor_id]) >= 20:
                    print(breakdownData)
                    self.producer.process(None, 'predictBreakdownEvent', breakdownData)
                    self.result_dir[sensor_id] = []
        else:
            self.data_dir[sensor_id].extend(message['data'])


if __name__ == '__main__':
    import os,sys
    folder = os.path.split(os.path.realpath(__file__))[0]
    sys.path += folder
    dataInit = DataInit()
    dataInit.start_run()
