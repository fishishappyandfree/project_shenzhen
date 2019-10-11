# coding=utf-8
import random

from KafkaService import KafkaProducerService, KafkaConsumerService

producer = KafkaProducerService()
consumer = KafkaConsumerService()

print(consumer.brokers)

print(producer.brokers)

#这里的两个参数，第一个是kafka的topic，替换成相应的topic，sensor就是那个sensor.avsc文件
for message in consumer.process('temp', 'sensor'):
#消费出来数据，处理...
    for i in message['data']:
        print(i,type(i))

