#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:28:23 2019

@author: hust
"""

import json, numpy, fastavro
import matplotlib.pyplot as plt

from io import BytesIO
from sqlalchemy import Column, Integer, LargeBinary, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SCHEMA_PATH = 'C:/Users/wzs/Desktop/sensor.avsc'
# sqlite://// + sqlite filepath
SQL_URL = 'sqlite:///D:/original_data/MT2_experiments_other/20190327200208_steel316_7000rpm_depth0.2_feed2000_width1/data.db'
# Capture Card ID / channel ID
SENSOR_ID = 'cDAQ9189-1D71297Mod3/ai0'
# open DB Browser for DQLite, and drag the file into the program, open "Browser Data"
START_TIME = '2019-03-27 20:03:00'
END_TIME = '2019-03-27 20:04:00'
# Sampling frequency
FS = 25600

Base = declarative_base()
metadata = Base.metadata


class SensorDatum(Base):
    __tablename__ = 'sensor_data'

    id = Column(Text(64), primary_key=True)
    create_time = Column(Text, nullable=False)
    sample_ts = Column(Integer, nullable=False)
    sensor_id = Column(Text, nullable=False)
    data = Column(LargeBinary, nullable=False)
    data_len = Column(Integer, nullable=False)

    def to_dict(self):
        return {c.name: getattr(self, c.name, None) for c in self.__table__.columns}

    Base.to_dict = to_dict


# engine = create_engine(SQL_URL, echo=True)
engine = create_engine(SQL_URL)
Session = sessionmaker(bind=engine)
session = Session()
schema = json.loads(open(SCHEMA_PATH, 'r').read())

signal_session = session.query(SensorDatum).filter(SensorDatum.sensor_id == SENSOR_ID,
                                                   SensorDatum.create_time.between(START_TIME, END_TIME)).all()

signal = []
for i in signal_session:
    # signal_dict = i.to_dict()
    with BytesIO(i.to_dict()['data']) as byte_io:
        # signal.append(fastavro.schemaless_reader(byte_io, schema)['data'])   # multiple dict object in a list
        signal.extend(fastavro.schemaless_reader(byte_io, schema)['data'])

signal = numpy.array(signal)  # transfer to numpy.ndarray
L = len(signal)

# PL = abs(numpy.fft.fft(signal / L))[: int(L / 2)]
PL = abs(numpy.fft.fft(signal))[: int(L / 2)]

PL[0] = 0
f = numpy.fft.fftfreq(L, 1)[: int(L / 2)]

plt.plot(f, PL)
plt.title('fft')

plt.show()

