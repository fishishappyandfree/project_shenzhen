# coding=utf-8
import io
import json
import os
import time

import avro.io
import avro.schema

from kafka import KafkaConsumer, KafkaProducer
from utils.common_config import get_common_config


class BaseKafkaService(object):
    """
    A Consumer for kafka that will produce json data
    provided for by converting from avro files
    """

    def __init__(self):
        common_config = get_common_config()
        self.brokers = common_config['kafka']['bootstrap_servers']
        self.max_poll_records = common_config['kafka']['max_poll_records']
        self.auto_offset_reset = common_config['kafka']['auto_offset_reset']
        self.enable_auto_commit = common_config['kafka']['enable_auto_commit']
        self.avro_path = common_config['kafka']['avro_path']
        try:
            iter(self.brokers)
        except TypeError:
            raise Exception('BROKERS must be a list')

        self.client_id = 'python-client-%s' % int(time.time())
        self.group_id = 'python-group-%s' % int(time.time())

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
            return avro.schema.Parse(open(avro_path, "rb").read().decode(encoding='utf-8'))
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

    def __init__(self):
        super(KafkaConsumerService, self).__init__()
        self.consumer = None

    def process(self, schema=None, *topic):
        """
        Iterator that yields json results and their message
        """
        self.consumer = KafkaConsumer(*topic,
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

    def __init__(self):
        super(KafkaProducerService, self).__init__()
        self.producer = KafkaProducer(bootstrap_servers=self.brokers)

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
