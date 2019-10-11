
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

SCHEMA_PATH = '/media/hust/f3fb8681-0cb6-4f0d-9e16-02d478f231c4/新建文件夹2/ae_analysis/ae_MT2/sensor.avsc'
# sqlite://// + sqlite filepath

SQL_URL_file__all_list = []
SQL_URL_folder_path='sqlite:////media/hust/f3fb8681-0cb6-4f0d-9e16-02d478f231c4/新建文件夹2/ae_analysis/ae_MT2_DATA/'

SQL_URL_file__normal_list = ['20190530095252_normal_al7075_6000rpm_feed2500_depth0.1_width3',
                             '20190530095837_normal_al7075_7000rpm_feed2500_depth0.1_width3',
                             '20190530103136_normal_al7075_8000rpm_feed2500_depth0.1_width3',
                             '20190530100250_normal_al7075_9000rpm_feed2500_depth0.1_width3',
                             '20190530100652_normal_al7075_10000rpm_feed2500_depth0.1_width3',
                             '20190530101043_normal_al7075_11000rpm_feed2500_depth0.1_width3',
                             '20190530101439_normal_al7075_12000rpm_feed2500_depth0.1_width3',
                             '20190530101839_normal_al7075_13000rpm_feed2500_depth0.1_width3',
                             '20190530102301_normal_al7075_14000rpm_feed2500_depth0.1_width3',
                             '20190530102700_normal_al7075_15000rpm_feed2500_depth0.1_width3']
SQL_URL_file__all_list.append(SQL_URL_file__normal_list)

SQL_URL_file__inner_list = ['20190423093510_inner_0.6_0.04_al7075_6000rpm_depth0.1_width3_feed2500',
                            '20190423093041_inner_0.6_0.04_al7075_7000rpm_depth0.1_width3_feed2500',
                            '20190423092628_inner_0.6_0.04_al7075_8000rpm_depth0.1_width3_feed2500',
                            '20190423092224_inner_0.6_0.04_al7075_9000rpm_depth0.1_width3_feed2500',
                            '20190423091821_inner_0.6_0.04_al7075_10000rpm_depth0.1_width3_feed2500',
                            '20190423091413_inner_0.6_0.04_al7075_11000rpm_depth0.1_width3_feed2500',
                            '20190423090857_inner_0.6_0.04_al7075_12000rpm_depth0.1_width3_feed2500',
                            '20190423090441_inner_0.6_0.04_al7075_13000rpm_depth0.1_width3_feed2500',
                            '20190423085938_inner_0.6_0.04_al7075_14000rpm_depth0.1_width3_feed2500',
                            '20190423085442_inner_0.6_0.04_al7075_15000rpm_depth0.1_width3_feed2500']
SQL_URL_file__all_list.append(SQL_URL_file__inner_list)

SQL_URL_file__outer_list = ['20190621151054_Outer-0.6-0.04_criticalG_al7075_6000rpm_feed2500_depth0.1_width3',
                            '20190621152546_Outer-0.6-0.04_criticalG_al7075_7000rpm_feed2500_depth0.1_width3',
                            '20190621152936_Outer-0.6-0.04_criticalG_al7075_8000rpm_feed2500_depth0.1_width3',
                            '20190621153441_Outer-0.6-0.04_criticalG_al7075_9000rpm_feed2500_depth0.1_width3',
                            '20190621153851_Outer-0.6-0.04_criticalG_al7075_10000rpm_feed2500_depth0.1_width3',
                            '20190621154331_Outer-0.6-0.04_criticalG_al7075_11000rpm_feed2500_depth0.1_width3',
                            '20190621154725_Outer-0.6-0.04_criticalG_al7075_12000rpm_feed2500_depth0.1_width3',
                            '20190621155120_Outer-0.6-0.04_criticalG_al7075_13000rpm_feed2500_depth0.1_width3',
                            '20190621155510_Outer-0.6-0.04_criticalG_al7075_14000rpm_feed2500_depth0.1_width3',
                            '20190621155858_Outer-0.6-0.04_criticalG_al7075_15000rpm_feed2500_depth0.1_width3']
SQL_URL_file__all_list.append(SQL_URL_file__outer_list)


save_folder_all_path = []
save_folder_path = '/media/hust/f3fb8681-0cb6-4f0d-9e16-02d478f231c4/新建文件夹2/ae_analysis/ae_MT2/ae_MT2_mat'

# 信号长度 2.0 1e08

#################切分标记格式################## 3.8 le7, 1.26 le8   (391400000,)
save_file_normal_list = ['normal_6000rpm',   # [0.5 1e8, 3.9 1e8]
                         'normal_7000rpm',   # [0.3 1e8, 3.9 1e8]
                         'normal_8000rpm',   # [0.5 1e8, 3.8 1e8]
                         'normal_9000rpm',   # [3.0 1e8, 4.0 1e8]
                         'normal_10000rpm',  # [0.3 1e8, 3.8 1e8]
                         'normal_11000rpm',  # [0.3 1e8, 4.0 1e8]
                         'normal_12000rpm',  # [2.0 1e8, 4.0 1e8]
                        #  'normal_13000rpm',  # [1.5 1e8, 4.0 1e8]   峰值非常接近量程
                        #  'normal_14000rpm',  # [0.6 1e8, 3.9 1e8]   峰值非常接近量程
                        #  'normal_15000rpm']  # [0.5 1e8, 3.9 1e8]   峰值非常接近量程
                        ]
save_folder_all_path.append(save_file_normal_list)

save_file_inner_list = ['inner_6000rpm',   # [0: -1]               全程都是有效数据，明显冲击特征，峰值非常接近量程
                        'inner_7000rpm',   # [0: -1]               全程都是有效数据，明显冲击特征，峰值非常接近量程
                        'inner_8000rpm',   # [0: -1]               全程都是有效数据，明显冲击特征，峰值非常接近量程
                        'inner_9000rpm',   # [0: -1]               全程都是有效数据，明显冲击特征，峰值非常接近量程
                        'inner_10000rpm',  # [0: -1]               全程都是有效数据，明显冲击特征，峰值超过量程
                        'inner_11000rpm',  # [0: -1]               全程都是有效数据，明显冲击特征，峰值超过量程
                        'inner_12000rpm',  # [0: -1]               全程都是有效数据，明显冲击特征，峰值超过量程
                        # 'inner_13000rpm',  # [0: -1]               全程都是有效数据，明显冲击特征，峰值超过量程
                        # 'inner_14000rpm',  # [0: -1]               全程都是有效数据，明显冲击特征，峰值超过量程（推测）
                        # 'inner_15000rpm']  # [0: -1]               全程都是有效数据，明显冲击特征，峰值超过量程（推测）
                        ]
save_folder_all_path.append(save_file_inner_list)

save_file_outer_list = ['outer_6000rpm',   # [0.09 1e9, 3.09 1e8]
                        'outer_7000rpm',   # [0.09 1e9, 3.95 1e8]
                        'outer_8000rpm',   # [0.5 1e8, -1]
                        'outer_9000rpm',   # [2.5 1e8, -1]
                        'outer_10000rpm',  # [1.5 1e8, -1]
                        'outer_11000rpm',  # [0.1 1e8, -1]
                        'outer_12000rpm',  # [0.2 1e8, -1]
                        # 'outer_13000rpm',  # [1.5 1e8, 4.0 1e8]  峰值非常接近量程
                        # 'outer_14000rpm',  # [0.1 1e8, 3.9 1e8]  峰值非常接近量程
                        # 'outer_15000rpm']  # [0.1 1e8, 3.91 1e8] 峰值非常接近量程
                        ]
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


for SQL_URL_file__fault_list, save_file_fault_list in zip(SQL_URL_file__all_list, save_folder_all_path):
    for SQL_URL_file_path, save_mat_file_path in zip(SQL_URL_file__fault_list, save_file_fault_list):
        read_data_save_mat(SQL_URL_file_path, save_mat_file_path)
