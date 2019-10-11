'''
该文件是用来辅助分析的, 观看每个数据文件, 画出时频域图, 找出有效数据分割点
'''
import os
from scipy import io
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def plot_time_fft(array, name):
    '''
    plot time and frequency picture
    :param array: one dimension array and class is numpy ndarry
    :param name: the array tag
    :return: no
    '''
    L1 = len(array)
    # plot time domain picture
    # plt.subplot(211)
    plt.plot(array)
    plt.title(name)
    PL = np.abs(np.fft.fft(array / L1))[: int(L1 / 2)]

    # plot frequency domain picture
    PL[0] = 0
    # PL2[0] = 0
    f1 = np.fft.fftfreq(L1, 1 / 25600)[: int(L1 / 2)]
   # f1 = np.fft.fftfreq(L1, 1 / 2000000)[: int(L1 / 2)]

    # plt.subplot(212)
    # plt.plot(f1, PL)
    # plt.title(name)
    # #plt.savefig(name+'.png', bbox_inches='tight')
    plt.show()

# see ae data
def load_mat_func(path):

    data = loadmat(path)
    data1 = data['MT1_x_feed_axis_data']
    return data1



def analysis(file_name, folder_path):
    print('='*50)
    print(file_name)
    data = load_mat_func(folder_path+file_name+'.mat')
    print(data.shape)  # (1, 1510400)
    plot_time_fft(data.flatten(), file_name)


MT1_x_feed_file__all_list = []
MT1_x_feed_folder_path='C:/Users/数据采集/Desktop/新建文件夹 (2)/ae_analysis/MT1/x_feed_axis/x_feed_axis_mat/'

normal_MT1_x_feed_file_list = ['MT1_x_normal_f_2000-9-36-26',  # 16000, 877500
                               'MT1_x_normal_f_3000-9-38-22',  # 65000, 545000
                               'MT1_x_normal_f_4000-9-38-59',  # 60000, 1140000
                               'MT1_x_normal_f_5000-9-39-55',  # 65000, 930000
                               'MT1_x_normal_f_6000-9-40-45',  # 55000, 775000
                               'MT1_x_normal_f_7000-9-41-26',  # 55000, 670000
                               'MT1_x_normal_f_8000-9-42-02',  # 90000, 630000
                               'MT1_x_normal_f_9000-9-42-41',  # 70000, 640000
                               'MT1_x_normal_f_10000-9-43-15']  # 160000, 730000
MT1_x_feed_file__all_list.append(normal_MT1_x_feed_file_list)

inner_MT1_x_feed_file_list = ['MT1_x_in_0.6_0.002_f_2000-13-25-54',  # 80000, 610000
                              'MT1_x_in_0.6_0.002_f_3000-13-26-54',  # 67000, 420000
                              'MT1_x_in_0.6_0.002_f_4000-13-27-19',  # 60000, 965000
                              'MT1_x_in_0.6_0.002_f_5000-13-28-01',  # 30000, 675000
                              'MT1_x_in_0.6_0.002_f_6000-13-28-49',  # 35000, 570000
                              'MT1_x_in_0.6_0.002_f_7000-13-29-27',  # 50000, 510000
                              'MT1_x_in_0.6_0.002_f_8000-13-29-57',  # 65000, 465000
                              'MT1_x_in_0.6_0.002_f_9000-13-30-43',  # 45000, 380000
                              'MT1_x_in_0.6_0.002_f_10000-13-31-03']  # 75000, 500000
MT1_x_feed_file__all_list.append(inner_MT1_x_feed_file_list)

outer_MT1_x_feed_file_list = ['MT1_x_out_0.6_0.002_f_2000-8-24-06',  # 100000, 630000
                              'MT1_x_out_0.6_0.002_f_3000-8-25-04',  # 45000, 390000
                              'MT1_x_out_0.6_0.002_f_4000-8-25-32',  # 45000, 850000
                              'MT1_x_out_0.6_0.002_f_5000-8-26-12',  # 45000, 690000
                              'MT1_x_out_0.6_0.002_f_6000-8-26-45',  # 50000, 585000
                              'MT1_x_out_0.6_0.002_f_7000-8-27-14',  # 40000, 500000
                              'MT1_x_out_0.6_0.002_f_8000-8-28-01',  # 25000, 425000
                              'MT1_x_out_0.6_0.002_f_9000-8-28-24',  # 40000, 395000
                              'MT1_x_out_0.6_0.002_f_10000-8-28-55']  # 40000, 470000
MT1_x_feed_file__all_list.append(outer_MT1_x_feed_file_list)

for i in MT1_x_feed_file__all_list:
    for j in i:
        analysis(j, MT1_x_feed_folder_path)








