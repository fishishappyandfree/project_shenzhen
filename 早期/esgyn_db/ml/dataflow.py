# coding=utf-8
import random

from KafkaService import KafkaProducerService, KafkaConsumerService
producer = KafkaProducerService()
consumer = KafkaConsumerService(auto_offset_reset='earliest',
                                enable_auto_commit=False)

print consumer.brokers

print producer.brokers
machine_no_arr = ['atm', 'G15JS0041', 'M03JS0042', 'M03JS0043', 'M03JS0044']
breakdown_arr = [76, 77, 78, 79]
i = 1
for message in consumer.process('temp', 'sensor'):
    breakdownMessage = {}
    percent = int(random.random() * 101)
    breakdownMessage['machineNo'] = machine_no_arr[int(random.random() * 5)]
    breakdownMessage['breakdownTypeId'] = breakdown_arr[int(random.random() * 4)]
    breakdownMessage['percent'] = percent
    breakdownMessage['created'] = message['sample_ts']


    if i % 10 == 0:
        producer.process(None, 'predictBreakdownEvent', breakdownMessage)
        producer.process('sensor', 'breakdownData', message)
        print i, '----->', percent     
    i = i + 1
  
 


