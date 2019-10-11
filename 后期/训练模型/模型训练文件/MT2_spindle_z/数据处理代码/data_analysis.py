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
    plt.subplot(211)
    plt.plot(array)
    plt.title(name)
    PL = np.abs(np.fft.fft(array / L1))[: int(L1 / 2)]

    # plot frequency domain picture
    PL[0] = 0
    # PL2[0] = 0
    f1 = np.fft.fftfreq(L1, 1 / 25600)[: int(L1 / 2)]

    plt.subplot(212)
    plt.plot(f1, PL)
    plt.title(name)
    plt.show()


def load_mat_func(path):
    """
    :param path: data path
    :return:  (6*2400, 2048),or (6*1200, 2048)
    """
    data = loadmat(path)
    data1 = data['spindle_z'][0]
    print(type(data))
    print(data['spindle_z'])
    print(type(data['spindle_z']))
    print(data['spindle_z'].shape)
    print(np.array(data['spindle_z']).shape)
    print(data1.shape)
    return data1


def analysis(folderpath, fault_list):
    '''

    :param folderpath: the folder path of data file
    :param fault_list: the various kind of fault data file
    :return: no
    '''
    data_list = []
    for i in fault_list:
        data_list.append(load_mat_func(folderpath+i))
        # plot_time_fft(load_mat_func(folderpath + i), i)
    return data_list

nomal_folderpath = '/home/hust/Desktop/JiangSu/MT2/Spindle_Program/Project/MT2/normal/mat_data/'
normal_unload_list = ['normal_unload_6000rpm-10-40-33.mat',
                      'normal_unload_9000rpm-10-52-31.mat',
                      'normal_unload_12000rpm-11-02-35.mat',
                      'normal_unload_15000rpm-11-14-23.mat']
analysis(nomal_folderpath, normal_unload_list)

normal_steel_list = ['normal_steel304_6000rpm_feed2500_depth0.1_width3-14-45-53.mat',
                     'normal_steel304_9000rpm_feed2500_depth0.1_width3-14-53-31.mat',
                     'normal_steel304_12000rpm_feed2500_depth0.1_width3-15-07-51.mat',
                     'normal_steel304_15000rpm_feed2500_depth0.1_width3-15-18-00.mat']
analysis(nomal_folderpath, normal_steel_list)

normal_al_list = ['normal_al7075_6000rpm_feed2500_depth0.1_width3-9-52-48.mat',
                  'normal_al7075_9000rpm_feed2500_depth0.1_width3-10-02-46.mat',
                  'normal_al7075_12000rpm_feed2500_depth0.1_width3-10-14-36.mat',
                  'normal_al7075_15000rpm_feed2500_depth0.1_width3-10-26-58.mat']
analysis(nomal_folderpath, normal_al_list)

inner_folderpath = '/home/hust/Desktop/wzs/MT2_data/inner_0.6_0.04/mat_data/'
inner_unload_list = ['20190422094004_inner_0.6_0.04_unload_6000rpm.mat',
                     '20190422095259_inner_0.6_0.04_unload_9000rpm.mat',
                     '20190422100513_inner_0.6_0.04_unload_12000rpm.mat',
                     '20190422101722_inner_0.6_0.04_unload_15000rpm.mat']
analysis(inner_folderpath, inner_unload_list)

inner_steel_list = ['20190424144412_inner_0.6_0.04_steel304_6000rpm_depth0.1_width3_feed2500.mat',
                    '20190424145121_inner_0.6_0.04_steel304_9000rpm_depth0.1_width3_feed2500.mat',
                    '20190424150254_inner_0.6_0.04_steel304_12000rpm_depth0.1_width3_feed2500.mat',
                    '20190424151432_inner_0.6_0.04_steel304_15000rpm_depth0.1_width3_feed2500.mat']
analysis(inner_folderpath, inner_steel_list)

inner_al_list = ['20190423093046_inner_0.6_0.04_al7075_7000rpm_depth0.1_width3_feed2500.mat',
                 '20190423092229_inner_0.6_0.04_al7075_9000rpm_depth0.1_width3_feed2500.mat',
                 '20190423091826_inner_0.6_0.04_al7075_10000rpm_depth0.1_width3_feed2500.mat',
                 '20190423085944_inner_0.6_0.04_al7075_14000rpm_depth0.1_width3_feed2500.mat']
analysis(inner_folderpath, inner_al_list)

outer_folderpath = '/home/hust/Desktop/wzs/MT2_data/outer_0.6_0.04_criticalG/mat_data/'
outer_unload_list = ['outer-0.6-0.04_criticalG_unload_6000rpm-11-03-23.mat',
                     'outer-0.6-0.04_criticalG_unload_9000rpm-11-20-46.mat',
                     'outer-0.6-0.04_criticalG_unload_12000rpm-11-38-05.mat',
                     'outer-0.6-0.04_criticalG_unload_15000rpm-11-55-16.mat']
analysis(outer_folderpath, outer_unload_list)

outer_steel_list = ['outer-0.6-0.04_criticalG_steel304_6000rpm_feed2500_depth0.1_width3-14-50-21.mat',
                    'outer-0.6-0.04_criticalG_steel304_9000rpm_feed2500_depth0.1_width3-14-56-35.mat',
                    'outer-0.6-0.04_criticalG_steel304_12000rpm_feed2500_depth0.1_width3-14-37-30.mat',
                    'outer-0.6-0.04_criticalG_steel304_15000rpm_feed2500_depth0.1_width3-14-47-00.mat']
analysis(outer_folderpath, outer_steel_list)

outer_al_list = ['outer-0.6-0.04_criticalG_al7075_6000rpm_feed2500_depth0.1_width3-15-10-49.mat',
                 'outer-0.6-0.04_criticalG_al7075_9000rpm_feed2500_depth0.1_width3-15-34-39.mat',
                 'outer-0.6-0.04_criticalG_al7075_12000rpm_feed2500_depth0.1_width3-15-47-22.mat',
                 'outer-0.6-0.04_criticalG_al7075_15000rpm_feed2500_depth0.1_width3-15-58-55.mat']
analysis(outer_folderpath, outer_al_list)


# 信号截取段
signal_split = {
    # normal
    'Normal':
    [
        # normal_unload_6000rpm  3个来回  每段250000
        [[200000, 450000], [800000, 1050000], [1400000, 1650000],
         [1900000, 2150000], [2600000, 2850000], [3500000, 3750000]],
        # normal_unload_9000rpm  3个来回  每段100000
        [[200000, 450000], [800000, 1050000], [1400000, 1650000],
         [1900000, 2150000], [2600000, 2850000], [3500000, 3750000]],
        # normal_unload_12000rpm  3个来回  每段90000
        [[200000, 450000], [800000, 1050000], [1400000, 1650000],
         [1900000, 2150000], [2600000, 2850000], [3500000, 3750000]],
        # normal_unload_15000rpm  3个来回  每段50000
        [[200000, 450000], [800000, 1050000], [1400000, 1650000],
         [1900000, 2150000], [2600000, 2850000], [3500000, 3750000]],

        # normal_steel304_6000rpm  3个来回  每段50000
        [[150000, 210000], [780000, 840000], [1410000, 1470000], [2040000, 2100000], [2570000, 2630000], [3090000, 3150000]],
        # normal_steel304_9000rpm  3个来回  每段40000
        [[305000, 365000], [830000, 890000], [1355000, 1415000], [1985000, 2045000], [2405000, 2465000], [3035000, 3095000]],
        # normal_steel304_12000rpm  3个来回  每段30000(只有5段,最后一段一分为二)
        [[565000, 625000], [980000, 1040000], [1405000, 1465000], [2040000, 2100000], [2560000, 2620000], [3300000, 3360000]],
        # normal_steel304_15000rpm  3个来回  每段30000
        [[375000, 435000], [795000, 855000], [1215000, 1275000], [1525000, 1585000], [2160000, 2220000], [3320000, 3380000]],
        
        # normal_al7075_6000rpm  3个来回  每段50000
        [[570000, 630000], [1120000, 1180000], [1535000, 1595000], [2095000, 2155000], [2920000, 2980000], [4445000, 4505000]],
        # normal_al7075_9000rpm  3个来回  每段40000
        [[1320000, 1380000], [2040000, 2100000], [2505000, 2565000], [3120000, 3180000], [3440000, 3500000], [4620000, 4680000]],
        # normal_al7075_12000rpm  3个来回  每段30000(只有5段,最后一段一分为二)
        [[530000, 590000], [1000000, 1060000], [1520000, 1580000], [2200000, 2260000], [3010000, 3070000], [4480000, 4540000]],
        # normal_al7075_15000rpm  3个来回  每段30000
        [[520000, 580000], [1230000, 1290000], [1900000, 1960000], [2720000, 2780000], [3230000, 3290000], [4785000, 4845000]],
    ],
    # Inner Raceway
    'Inner_Raceway':
    [
        # inner_0.6_0.04_unload_6000rpm  3个来回  每段250000
        [[400000, 650000], [1000000, 1250000], [1600000, 1850000],
         [2100000, 2350000], [2800000, 3050000], [3700000, 3950000]],
        # inner_0.6_0.04_unload_9000rpm  3个来回  每段100000
        [[400000, 650000], [1000000, 1250000], [1600000, 1850000],
         [2100000, 2350000], [2800000, 3050000], [3700000, 3950000]],
        # inner_0.6_0.04_unload_12000rpm  3个来回  每段100000
        [[500000, 750000], [1100000, 1350000], [1700000, 1950000],
         [2300000, 2550000], [2900000, 3150000], [3700000, 3950000]],
        # inner_0.6_0.04_unload_15000rpm  3个来回  每段50000
        [[400000, 650000], [1000000, 1250000], [1600000, 1850000],
         [2100000, 2350000], [2800000, 3050000], [3400000, 3650000]],

        # inner_0.6_0.04_steel304_6000rpm  3个来回  每段50000
        [[800000, 1050000], [1300000, 1550000], [1800000, 2050000],
         [2300000, 2550000], [2800000, 3050000], [3400000, 3650000]],
        # inner_0.6_0.04_steel304_9000rpm  3个来回  每段40000
        [[400000, 650000], [1000000, 1250000], [1600000, 1850000],
         [2100000, 2350000], [2600000, 2850000], [3000000, 3250000]],
        # inner_0.6_0.04_steel304_12000rpm  3个来回  每段30000(只有5段,最后一段一分为二)
        [[400000, 650000], [1000000, 1250000], [1600000, 1850000],
         [2100000, 2350000], [2600000, 2850000], [3000000, 3250000]],
        # inner_0.6_0.04_steel304_15000rpm  3个来回  每段30000
        [[400000, 650000], [1000000, 1250000], [1600000, 1850000],
         [2100000, 2350000], [2600000, 2850000], [3000000, 3250000]],
        
        # inner_0.6_0.04_al7075_7000rpm  3个来回  每段250000
        [[400000, 650000], [1000000, 1250000], [1600000, 1850000],
         [2100000, 2350000], [2800000, 3050000], [3700000, 3950000]],
        # inner_0.6_0.04_al7075_9000rpm  3个来回  每段10000
        [[400000, 650000], [1000000, 1250000], [1600000, 1850000],
         [2100000, 2350000], [2800000, 3050000], [3700000, 3950000]],
        # inner_0.6_0.04_al7075_10000rpm  3个来回  每段100000
        [[400000, 650000], [1000000, 1250000], [1600000, 1850000],
         [2100000, 2350000], [2800000, 3050000], [3700000, 3950000]],
        # inner_0.6_0.04_al7075_14000rpm  3个来回  每段50000
        [[800000, 1050000], [1500000, 1750000], [2100000, 2350000],
         [2600000, 2850000], [3200000, 3450000], [3900000, 4150000]],
    ],
    # Outer Raceway
    "Outer_Raceway":
[
        # outer-0.6-0.04_criticalG_unload_6000rpm  3个来回  每段250000
        [[400000, 650000], [1000000, 1250000], [1600000, 1850000],
         [2100000, 2350000], [2800000, 3050000], [3700000, 3950000]],
        # outer-0.6-0.04_criticalG_unload_9000rpm  3个来回  每段10000
        [[400000, 650000], [1000000, 1250000], [1600000, 1850000],
         [2100000, 2350000], [2800000, 3050000], [3700000, 3950000]],
        # outer-0.6-0.04_criticalG_unload_12000rpm  3个来回  每段100000
        [[1200000, 1450000], [1800000, 2050000], [2600000, 2850000],
         [3200000, 3450000], [4000000, 4200000], [4800000, 5000000]],
        # outer-0.6-0.04_criticalG_unload_15000rpm  3个来回  每段50000
        [[1200000, 1450000], [1800000, 2050000], [2600000, 2850000],
         [3200000, 3450000], [4000000, 4200000], [4800000, 5000000]],

        # outer-0.6-0.04_criticalG_steel304_6000rpm  3个来回  每段50000
        [[528000, 588000], [845000, 905000], [1370000, 1430000], [1895000, 1955000], [2630000, 2690000], [3575000, 3635000]],
        # outer-0.6-0.04_criticalG_steel304_9000rpm  3个来回  每段40000
        [[465000, 525000], [885000, 945000], [1408000, 1468000], [1828000, 1888000], [2565000, 2625000], [3508000, 3568000]],
        # outer-0.6-0.04_criticalG_steel304_12000rpm  3个来回  每段30000(只有5段,最后一段一分为二)
        [[380000, 440000], [695000, 755000], [1425000, 1485000], [1640000, 1700000], [2060000, 2120000], [3330000, 3390000]],
        # outer-0.6-0.04_criticalG_steel304_15000rpm  3个来回  每段30000
        [[365000, 425000], [995000, 1055000], [1420000, 1480000], [1835000, 1895000], [2365000, 2425000], [3095000, 3155000]],
        
        # outer-0.6-0.04_criticalG_al7075_6000rpm  3个来回  每段50000
        [[1200000, 1260000], [1775000, 1835000], [2635000, 2695000], [2920000, 2980000], [4070000, 4130000], [4795000, 4855000]],
        # outer-0.6-0.04_criticalG_al7075_9000rpm  3个来回  每段40000
        [[480000, 540000], [1200000, 1260000], [1780000, 1840000], [2485000, 25450000], [3355000, 3415000], [4515000, 4575000]],
        # outer-0.6-0.04_criticalG_al7075_12000rpm  3个来回  每段30000(只有5段,最后一段一分为二)
        [[400000, 460000], [1015000, 1075000], [1630000, 16900000], [2330000, 2390000], [2820000, 2880000], [4085000, 4145000]],
        # outer-0.6-0.04_criticalG_al7075_15000rpm  3个来回  每段30000
        [[300000, 360000], [810000, 870000], [1340000, 1400000], [2060000, 2120000], [3580000, 3640000], [4360000, 4420000]],
    ]
}




