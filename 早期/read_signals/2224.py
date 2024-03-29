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
SQL_URL = 'sqlite:///D:/original_data/MT3_experiments_other/20190402191859_x_0.1-0.02_new_tool_7000rpm_feed1200_depth0.4_width2/data.db'
# Capture Card ID / channel ID
SENSOR_ID = 'cDAQ9189-1D71297Mod6/ai2'
# open DB Browser for DQLite, and drag the file into the program, open "Browser Data"
START_TIME = '2019-04-02 19:19:00'
END_TIME = '2019-04-02 19:20:00'
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
PL = abs(numpy.fft.fft(signal))[320000:360000]

PL[0] = 0
f = numpy.fft.fftfreq(L, 1)[320000:360000]

plt.plot(f, PL)
plt.title('fft')

plt.show()

