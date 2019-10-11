
'''=====================读取机床二unload/al/steel的.db 文件========================='''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:28:23 2019

@author: hust
"""

import json, numpy, fastavro
import matplotlib.pyplot as plt

from scipy import io
from io import BytesIO
from sqlalchemy import Column, Integer, LargeBinary, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import matplotlib as mpl

SCHEMA_PATH = 'sensor.avsc'
# sqlite://// + sqlite filepath

SQL_URL_file__all_list = []
SQL_URL_folder_path=[]
SQL_URL_normal_folder_path='sqlite:///G:/MT2_experiments_ae/'
SQL_URL_inner_folder_path='sqlite:///G:/MT2_experiments_ae/'
SQL_URL_outer_folder_path='sqlite:///J:/MT2_ae/Outer-0.6-0.04_criticalG/'

SQL_URL_folder_path.append(SQL_URL_normal_folder_path)
SQL_URL_folder_path.append(SQL_URL_inner_folder_path)
SQL_URL_folder_path.append(SQL_URL_outer_folder_path)


SQL_URL_file__normal_list = ['20190529105550_normal_unload_10000rpm',  # normal unload
                             '20190530100652_normal_al7075_10000rpm_feed2500_depth0.1_width3',  # normal al
                             '20190529145910_normal_steel304_10000rpm_feed2500_depth0.1_width3']  # normal steel
SQL_URL_file__all_list.append(SQL_URL_file__normal_list)

SQL_URL_file__inner_list = ['20190422095618_inner_0.6_0.04_unload_10000rpm',  # inner unload
                            '20190423091821_inner_0.6_0.04_al7075_10000rpm_depth0.1_width3_feed2500',  # inner al
                            '20190424145523_inner_0.6_0.04_steel304_10000rpm_depth0.1_width3_feed2500']  # inner steel
SQL_URL_file__all_list.append(SQL_URL_file__inner_list)

SQL_URL_file__outer_list = ['20190621112618_Outer-0.6-0.04_criticalG_unload_10000rpm',  # outer unload
                            '20190621153851_Outer-0.6-0.04_criticalG_al7075_10000rpm_feed2500_depth0.1_width3',  # outer al
                            '20190621145938_Outer-0.6-0.04_criticalG_steel304_10000rpm_feed2500_depth0.1_width3']  # outer steel

SQL_URL_file__all_list.append(SQL_URL_file__outer_list)


save_folder_all_path = []
save_folder_path = 'C:/Users/数据采集/Desktop/新建文件夹 (2)/ae_analysis/ae_MT2/ae_MT2_mat/'

#################切分标记格式################## 3.8 le7, 1.26 le8   (391400000,)
save_file_normal_list = ['unload_normal_10000rpm',
                         'al_normal_10000rpm',  #
                         'steel_normal_10000rpm']  #
save_folder_all_path.append(save_file_normal_list)

save_file_inner_list = ['unload_inner_10000rpm',
                        'al_inner_10000rpm',
                        'steel_inner_10000rpm']  #
save_folder_all_path.append(save_file_inner_list)

save_file_outer_list = ['unload_outer_10000rpm',
                        'al_outer_10000rpm',
                        'steel_outer_10000rpm']  #
save_folder_all_path.append(save_file_outer_list)

SENSOR_ID = 'cDAQ9189-1D91958Mod5/ai1'  # MT2 声发射

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

def read_data_save_mat(fault_folder_path, SQL_URL, savepath):
    # engine = create_engine(SQL_URL, echo=True)
    engine = create_engine(fault_folder_path+SQL_URL+'/data.db')
    Session = sessionmaker(bind=engine)
    session = Session()
    schema = json.loads(open(SCHEMA_PATH, 'r').read())
    signal_session = session.query(SensorDatum).filter(SensorDatum.sensor_id == SENSOR_ID,).all()
    signal = []
    for i in signal_session:
        with BytesIO(i.to_dict()['data']) as byte_io:
            signal.extend(fastavro.schemaless_reader(byte_io, schema)['data'])

    signal = numpy.array(signal)  # transfer to numpy.ndarray
    print('='*60)
    print(signal.shape)
    print(SQL_URL)
    print(savepath)
    plot_time_fft(signal, savepath)
    # 保存数据
   # signal_save={
    #    'signal_save' : signal
   # }
   # io.savemat(save_folder_path+savepath, signal_save)

def plot_time_fft(signal, picture_name):

    #signal = signal[200000:500000]
    # L = len(signal)
    #
    # PL = abs(numpy.fft.fft(signal / L))[: int(L / 2)]
    # PL[0] = 0
    # f = numpy.fft.fftfreq(L, 1 / FS)[: int(L / 2)]
    #
    plt.figure(figsize=(13, 6))

    # plt.subplot(211)
    #mpl.rcParams['agg.path.chunksize'] = 10000
    plt.plot(signal)  # (171600000,)
    plt.title(picture_name+'time')
    #print(plt.ylim([-0.5,0.5]))

    #plt.subplot(212)

    # plt.plot(f, PL)
    # plt.title(picture_name+'fft')
    plt.show()


for fault_folder_path,SQL_URL_file__fault_list, save_file_fault_list in zip(SQL_URL_folder_path,SQL_URL_file__all_list, save_folder_all_path):
    for SQL_URL_file_path, save_mat_file_path in zip(SQL_URL_file__fault_list, save_file_fault_list):
        read_data_save_mat(fault_folder_path, SQL_URL_file_path, save_mat_file_path)
