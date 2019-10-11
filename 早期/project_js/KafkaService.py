# coding=utf-8
import io
import json
import os
import time

import avro.io
import avro.schema

from kafka import KafkaConsumer, KafkaProducer


class BaseKafkaService(object):
    """
    A Consumer for kafka that will produce json data
    provided for by converting from avro files
    """

    def __init__(self, **config):
        self.avro_path = config.get('avro_path', '')
        #self.brokers = config.get('bootstrap_servers', ['10.143.62.181:9092','10.143.62.191:9092'])
        self.brokers = config.get('bootstrap_servers', ['10.143.62.232:9092'])
        self.max_poll_records = config.get('max_poll_records', 10)
        self.auto_offset_reset = config.get('auto_offset_reset', 'latest') # latest，且下面改为False，则表示每次启动消费端都是从最新的生产节点开始消费
        self.enable_auto_commit = config.get('enable_auto_commit', False)  # 若是True，则每次启动消费端，都是从以前的生产节点消费，False，从最新的生产开始
        try:
            iter(self.brokers)
        except TypeError:
            raise Exception('BROKERS must be a list')

        self.client_id = config.get('client_id', 'client-%s' % int(time.time()))
        self.group_id = config.get('group_id', 'group-%s' % int(time.time()))

    @staticmethod
    def _decode(msg, schema):
        if schema is None:
            #
            # No schema defined must be plain json
            #
            return msg

        bytes_reader = io.BytesIO(msg.value)
        decoder = avro.io.BinaryDecoder(bytes_reader)
        reader = avro.io.DatumReader(schema)
        return reader.read(decoder)

    def _schema(self, schema=None):
        """
        Read the passed in schema according to the path
        """
        if self.avro_path is None:
            return None

        schema = '%s.avsc' % schema
        avro_path = os.path.join(self.avro_path, schema)
        

        if os.path.exists(avro_path):
            return avro.schema.Parse(open(avro_path, "rb").read().decode())
        return None

    def _writer(self, schema):
        schema = self._schema(schema)
        if schema is None:
            return None, None, None

        writer = avro.io.DatumWriter(schema)
        bytes_writer = io.BytesIO()
        encoder = avro.io.BinaryEncoder(bytes_writer)
        return writer, bytes_writer, encoder


class KafkaConsumerService(BaseKafkaService):
    """
    Consumer, binary avro data or json data
    """

    def __init__(self, **kwargs):
        super(KafkaConsumerService, self).__init__(**kwargs)
        self.consumer = None

    def process(self, topic, schema=None):
        """
        Iterator that yields json results and their message
        """
        self.consumer = KafkaConsumer(topic,
                                      client_id=self.client_id,
                                      group_id=self.group_id,
                                      auto_offset_reset=self.auto_offset_reset,
                                      enable_auto_commit=self.enable_auto_commit,
                                      bootstrap_servers=self.brokers,
                                      max_poll_records=self.max_poll_records)

        schema = self._schema(schema=schema)

        for message in self.consumer:
            if schema:
                avro_msg = self._decode(msg=message, schema=schema)
            else:
                avro_msg = message.value

            yield avro_msg


class KafkaProducerService(BaseKafkaService):
    """
    Producer, binary avro data or json data
    """

    def __init__(self, **kwargs):
        super(KafkaProducerService, self).__init__(**kwargs)
        #self.producer = KafkaProducer(bootstrap_servers=['10.143.62.181:9092','10.143.62.191:9092'])
        self.producer = KafkaProducer(bootstrap_servers=['10.143.62.232:9092'])

    def _transmit(self, topic, raw_bytes):
        return self.producer.send(topic, raw_bytes)

    def process(self, schema, topic, data):
        """
        json data or binary avro data
        """
        writer, bytes_writer, encoder = self._writer(schema)
        if writer is None and bytes_writer is None and encoder is None:
            raw_bytes = json.dumps(data).encode(encoding='utf-8')
        else:
            writer.write(data, encoder)
            raw_bytes = bytes_writer.getvalue()

        return self._transmit(topic, raw_bytes)

    def close(self):
        self.producer.flush()
        self.producer.close()
