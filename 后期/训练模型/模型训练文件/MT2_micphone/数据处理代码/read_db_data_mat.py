"""This code is currently used only on linux."""

import os
import json
import sqlite3
import fastavro
import numpy as np
from io import BytesIO
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, LargeBinary, Text, create_engine

# from scipy.fftpack import fft
from scipy.io import savemat
# import matplotlib.pyplot as plt


class SensorDatum(declarative_base()):

    __tablename__ = 'sensor_data'

    id = Column(Text(64), primary_key=True)
    create_time = Column(Text, nullable=False)
    sample_ts = Column(Integer, nullable=False)
    sensor_id = Column(Text, nullable=False)
    data = Column(LargeBinary, nullable=False)
    data_len = Column(Integer, nullable=False)

    def to_dict(self):
        return {c.name: getattr(self, c.name, None) for c in self.__table__.columns}

    declarative_base().to_dict = to_dict


def get_summary(fn_db_file_path):

    conn = sqlite3.connect(fn_db_file_path)
    c = conn.cursor()
    c.execute("select * from 'sensor_data'")

    all_rows = c.fetchall()
    fn1_start_time = all_rows[0][1].split('.')[0]  # 'create_time' of first row
    fn1_end_time = all_rows[-1][1].split('.')[0]  # 'create_time' of final row

    conn.close()

    schema_path = 'sensor.avsc'

    # format: 'sqlite://// + .db file path' (absolute path required)
    sql_url = 'sqlite:////' + fn_db_file_path

    engine = create_engine(sql_url)
    fn1_session = sessionmaker(bind=engine)
    fn1_schema = json.loads(open(schema_path, 'r').read())

    return fn1_start_time, fn1_end_time, fn1_session(), fn1_schema


def get_data(sensor_id):
    signal_session = session.query(SensorDatum).filter(SensorDatum.sensor_id == sensor_id,
                                                       SensorDatum.create_time.between(start_time, end_time)).all()
    signal = []
    for i in signal_session:
        with BytesIO(i.to_dict()['data']) as byte_io:
            signal.extend(fastavro.schemaless_reader(byte_io, schema)['data'])

    signal = np.array(signal)

    return signal


# --------------------------------  single file processing  -------------------------------------------------
# MT3_data_path =  '/home/logan/win7/Spindle_Program/Projects/Data_Repository/' \
#                   'MT_other/MT3_experiments_other/20190402191430_x_0.1-0.02_new_tool_7000rpm_feed1500_depth0.4_width2/data.db'
# spindle_x_id = 'cDAQ9189-1D71297Mod5/ai0'
# spindle_y_id = 'cDAQ9189-1D71297Mod5/ai1'
# spindle_z_id = 'cDAQ9189-1D71297Mod5/ai2'
# feed_x_id = 'cDAQ9189-1D71297Mod5/ai3'
# feed_y_id = 'cDAQ9189-1D71297Mod4/ai1'
#
# start_time, end_time, session, schema = get_summary(MT3_data_path)
#
# spindle_x = get_data(spindle_x_id)
# spindle_y = get_data(spindle_y_id)
# spindle_z = get_data(spindle_z_id)
# feed_x = get_data(feed_x_id)
# feed_y = get_data(feed_y_id)
#
# plt.subplot(211)
# plt.plot(feed_x[25600:51200])
# plt.subplot(212)
# plt.plot(abs(fft(feed_x[25600:51200])))
# plt.tight_layout()
# plt.show()
#
# # savemat('x_0.1-0.02_new_tool_7000rpm_feed1500_depth0.4_width2.mat',
# #         dict(sp_x=spindle_x, sp_y=spindle_y, sp_z=spindle_z, fd_x=feed_x, fd_y=feed_y))
# print('done!')


# -------------------------------  batch file processing  ---------------------------------------------------
fault_directory = '/home/hust/Desktop/JiangSu/data_from_MT2/Spindle_Program/Project/MT2/inner_0.6_0.04/'
save_directory = '/home/hust/Desktop/wzs/MT2_micphone/MT2_micphone_data/inner_0.6_0.04/'
# MT1 spindle and feed axis channels
# spindle_x_id = 'cDAQ9189-1D71297Mod1/ai0'
# spindle_y_id = 'cDAQ9189-1D71297Mod1/ai1'
# spindle_z_id = 'cDAQ9189-1D71297Mod1/ai2'
# feed_x_id = 'cDAQ9189-1D71297Mod1/ai3'
# feed_y_id = 'cDAQ9189-1D71297Mod2/ai3'

# MT2 spindle channels
# spindle_x_id = 'cDAQ9189-1D71297Mod3/ai0'
# spindle_y_id = 'cDAQ9189-1D71297Mod3/ai1'
# spindle_z_id = 'cDAQ9189-1D71297Mod3/ai2'
# feed_x_id = 'cDAQ9189-1D71297Mod3/ai3'
# feed_y_id = 'cDAQ9189-1D71297Mod4/ai0'
MT2_mic_id = 'cDAQ9189-1D71297Mod6/ai1'

# MT3 spindle and feed axis channels
# spindle_x_id = 'cDAQ9189-1D71297Mod5/ai0'
# spindle_y_id = 'cDAQ9189-1D71297Mod5/ai1'
# spindle_z_id = 'cDAQ9189-1D71297Mod5/ai2'
# feed_x_id = 'cDAQ9189-1D71297Mod5/ai3'
# feed_y_id = 'cDAQ9189-1D71297Mod4/ai1'

# rule out those file_names which point not to directories
file_name_list = [name for name in os.listdir(fault_directory + 'raw_data/')
                  if os.path.isdir(fault_directory + 'raw_data/' + name)]
num_files = len(file_name_list)

print('{} files in chosen directory.'.format(num_files))

for file_index in range(num_files):

    file_name = file_name_list[file_index]

    output_name = save_directory + file_name + '.mat'

    if not os.path.exists(output_name):

        db_path = fault_directory + 'raw_data/' + file_name + '/data.db'
        start_time, end_time, session, schema = get_summary(db_path)

        # spindle_x = get_data(spindle_x_id)
        # spindle_y = get_data(spindle_y_id)
        # spindle_z = get_data(spindle_z_id)
        # MT2_feed_x = get_data(feed_x_id)
        # MT2_feed_y = get_data(feed_y_id)
        MT2_mic = get_data(MT2_mic_id)
        # if not os.path.exists(fault_directory + 'mat_data'):
        #     os.makedirs(fault_directory + 'mat_data')

        savemat(output_name, dict(MT2_mic=MT2_mic))
        print('done processing file {} in {}.'.format(file_index+1, num_files))

    else:
        print('file {} already converted'.format(file_index+1))
