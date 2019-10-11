# coding=utf-8
import random
import collections
import numpy
import itertools
from run_model_gpu import Model
from KafkaService import KafkaProducerService, KafkaConsumerService


class DataInit(object):
    def __init__(self, **kwargs):
        self.producer = KafkaProducerService()
        self.consumer = KafkaConsumerService()
        print(self.consumer.brokers)
        print(self.producer.brokers)
        self.cache_deque = collections.deque()


    def addData(self,):
        #这里的两个参数，第一个是kafka的topic，替换成相应的topic，sensor就是那个sensor.avsc文件
        for message in self.consumer.process('temp', 'sensor'):
            #消费出来数据，处理...
            #print message['data']
            for i in message['data']:
                #print i,type(i) 
                self.cache_deque.append(i)
                 
                 
    def get_data_for_predict(self, buffer, count):
        if len(buffer) < count:
            return
        result = []
        #with self._lock:
        for i in itertools.islice(buffer, count):
            result.append(i)
        return tuple(result)


    def start_run(self):
        self.model = Model()
        for message in self.consumer.process('temp', 'sensor'):
        #消费出来数据，处理...
        #print message['data']
            for i in message['data']:
                #print(i,type(i))
                self.cache_deque.append(i)
                result = self.get_data_for_predict(self.cache_deque,2048)
                #print(result)
                #print(type(result[0]))
                if result:
                    print('*'*50)
                    print("result:", type(result))  # tuple 类型
                    print(numpy.array(result).shape) # （2048,）
                    self.model.run_test(result)
                else:
                    continue



# producer = KafkaProducerService()
# consumer = KafkaConsumerService()
# print(consumer.brokers)
# print(producer.brokers)
dataInit = DataInit()
dataInit.start_run()