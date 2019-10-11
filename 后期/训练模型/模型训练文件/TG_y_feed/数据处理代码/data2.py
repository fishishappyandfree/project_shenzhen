
'''=====================读取机床二的.db 文件========================='''
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
# SQL_URL_folder_path='sqlite:///C:/Users/数据采集/Desktop/新建文件夹 (2)/ae_analysis/TG/TG_y_feed/TG_y_feed_data/'
SQL_URL_folder_path='sqlite:///TG_y_feed_data/'


# C:/Users/数据采集/Desktop/新建文件夹 (2)/ae_analysis/TG/TG_y_feed/TG_y_feed_data/


SQL_URL_file__normal_list = ['20190514090941_new_new_feed_Y_1000_num1',
                             # '20190514090824_new_new_feed_Y_2000_num1',
                             # '20190514090717_new_new_feed_Y_3000_num1',
                             # '20190514090624_new_new_feed_Y_4000_num1',
                             # '20190514090455_new_new_feed_Y_5000_num1',
                             '20190514090313_new_new_feed_Y_6000_num1',
                             '20190514090218_new_new_feed_Y_7000_num1',
                             '20190514090126_new_new_feed_Y_8000_num1']
SQL_URL_file__all_list.append(SQL_URL_file__normal_list)

SQL_URL_file__inner_list = ['20190517145301_Y_feed_axis_inner_race_0.6_0.02_feed_Y_1000_num2',
                            # '20190517145154_Y_feed_axis_inner_race_0.6_0.02_feed_Y_2000_num2',
                            # '20190517145114_Y_feed_axis_inner_race_0.6_0.02_feed_Y_3000_num2',
                            # '20190517145003_Y_feed_axis_inner_race_0.6_0.02_feed_Y_4000_num2',
                            # '20190517144921_Y_feed_axis_inner_race_0.6_0.02_feed_Y_5000_num2',
                            '20190517144817_Y_feed_axis_inner_race_0.6_0.02_feed_Y_6000_num2',
                            '20190517144714_Y_feed_axis_inner_race_0.6_0.02_feed_Y_7000_num2',
                            '20190517144619_Y_feed_axis_inner_race_0.6_0.02_feed_Y_8000_num2']
SQL_URL_file__all_list.append(SQL_URL_file__inner_list)

SQL_URL_file__outer_list = ['20190520141741_Y_feed_axis_outer_race_0.6_0.02_feed_Y_1000_num1',
                            # '20190520141643_Y_feed_axis_outer_race_0.6_0.02_feed_Y_2000_num1',
                            # '20190520141523_Y_feed_axis_outer_race_0.6_0.02_feed_Y_3000_num1',
                            # '20190520141433_Y_feed_axis_outer_race_0.6_0.02_feed_Y_4000_num1',
                            # '20190520141321_Y_feed_axis_outer_race_0.6_0.02_feed_Y_5000_num1',
                            '20190520141224_Y_feed_axis_outer_race_0.6_0.02_feed_Y_6000_num1',
                            '20190520141146_Y_feed_axis_outer_race_0.6_0.02_feed_Y_7000_num1',
                            '20190520141019_Y_feed_axis_outer_race_0.6_0.02_feed_Y_8000_num1']
SQL_URL_file__all_list.append(SQL_URL_file__outer_list)


save_folder_all_path = []
save_folder_path = 'TG_y_feed_mat/'

#################切分标记格式################## 3.8 le7, 1.26 le8   (391400000,)
save_file_normal_list = ['normal_feed_1000',   # [280000, 2050000]
                         # 'normal_feed_2000',   # [300000, 1160000]
                         #'normal_feed_3000',   # [280000, 800000]
                         #'normal_feed_4000',   # [] space
                         #'normal_feed_5000',   # [200000, 550000]
                         'normal_feed_6000',
# [[200000, 240000], [257000, 297000], [365000, 405000], [365000, 405000], [418000, 458000], [472000, 512000]]
                         'normal_feed_7000',   # [215000, 480000]
                         'normal_feed_8000']
# [[516000, 546000], [558000, 588000], [558000, 588000], [600000, 630000], [643000, 673000], [685000, 715000]]
save_folder_all_path.append(save_file_normal_list)

save_file_inner_list = ['inner_feed_1000',   # [120000, 1821000]
                        # 'inner_feed_2000',   # [130000, 1040000]
                        #'inner_feed_3000',   # space
                        #'inner_feed_4000',   # [194000, 645000]
                        #'inner_feed_5000',   # [] space
                        'inner_feed_6000',   # [90000, 400000]
                        'inner_feed_7000',
# [[110000, 140000], [160000, 190000], [205000, 235000], [255000, 285000], [300000, 330000], [350000, 380000]]
                        'inner_feed_8000']
# [[192000, 222000], [233000, 263000], [275000, 305000], [317000, 347000], [317000, 347000], [358000, 388000]]
save_folder_all_path.append(save_file_inner_list)

save_file_outer_list = ['outer_feed_1000',   # [600000, 1900000]
                        # 'outer_feed_2000',   # space
                        #'outer_feed_3000',   # [290000, 880000]
                        #'outer_feed_4000',   #
                        #'outer_feed_5000',   #
                        'outer_feed_6000',
# [[190000, 230000], [243500, 283500], [298000, 338000], [298000, 338000], [405000, 445000], [460000, 500000]]
                        'outer_feed_7000',
# [[155000, 185000], [200000, 230000], [250000, 280000], [295000, 325000], [342000, 372000], [390000, 420000]]
                        'outer_feed_8000']
# [[128000, 158000], [170000, 200000], [170000, 200000], [213000, 243000], [254000, 284000], [296000, 326000]]
save_folder_all_path.append(save_file_outer_list)

SENSOR_ID = 'cDAQ2Mod2/ai1'  # TG 磨床y_feed
# SENSOR_ID = 'cDAQ2Mod2/ai0'  # TG 磨床x_feed# Sampling frequency
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

def read_data_save_mat(SQL_URL, savepath):
    # engine = create_engine(SQL_URL, echo=True)
    engine = create_engine(SQL_URL_folder_path+SQL_URL+'/data.db')
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
    print(signal.shape)  # (2624000,)
    print(SQL_URL)
    print(savepath)
    # plot_time_fft(signal, savepath)
    # 保存数据
    signal_save={
       'TG_y_feed' : signal
    }
    io.savemat(save_folder_path+savepath+'.mat', signal_save)

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


for SQL_URL_file__fault_list, save_file_fault_list in zip(SQL_URL_file__all_list, save_folder_all_path):
    for SQL_URL_file_path, save_mat_file_path in zip(SQL_URL_file__fault_list, save_file_fault_list):
        read_data_save_mat(SQL_URL_file_path, save_mat_file_path)
