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
    data1 = data['MT2_mic']
    return data1



def analysis(file_name, folder_path):
    print('='*50)
    print(file_name)
    data = load_mat_func(folder_path+file_name)
    print(data.shape)  # (1, 1510400)
    plot_time_fft(data.flatten(), file_name)


MT2_mic_normal_folder_path = '/home/hust/Desktop/wzs/MT2_micphone/MT2_micphone_data/normal_slightG/'
normal_data_file_list = [['noamal_slightG_unload_6000rpm-15-48-44.mat',  #[2000000, 6500000]
                          'noamal_slightG_unload_9000rpm-16-05-17.mat',  # [500000,5000000]
                          'noamal_slightG_unload_12000rpm-16-20-29.mat', #[1000000, 6000000]
                          'noamal_slightG_unload_15000rpm-16-36-04.mat'], #[2000000, 6500000]
                         ['noamal_slightG_al7075_6000rpm_feed2500_depth0.1_width3-9-10-19.mat', #[1500000, 5500000]
                          'noamal_slightG_al7075_9000rpm_feed2500_depth0.1_width3-9-19-53.mat', #[1000000, 4000000]
                          'noamal_slightG_al7075_12000rpm_feed2500_depth0.1_width3-9-32-23.mat', #[2000000, 4000000]
                          'noamal_slightG_al7075_15000rpm_feed2500_depth0.1_width3-9-45-48.mat'], #[1000000, 5000000]
                         ['noamal_slightG_steel304_6000rpm_feed2500_depth0.1_width3-16-45-14.mat', #[500000, 3500000]
                          'noamal_slightG_steel304_9000rpm_feed2500_depth0.1_width3-16-51-27.mat', # [500000, 3500000]
                          'noamal_slightG_steel304_12000rpm_feed2500_depth0.1_width3-17-03-05.mat', #[500000, 3500000]
                          'noamal_slightG_steel304_15000rpm_feed2500_depth0.1_width3-17-12-18.mat']] #[500000, 3000000]

for i in normal_data_file_list:
    for j in i:
        analysis(j, MT2_mic_normal_folder_path)








