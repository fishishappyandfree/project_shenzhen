"""
【数据预处理思路】
1 每条数据，截取6段出来， 对每段进行数据增强（6段之间有断开，不能拼接起来再进行滑窗数据增强）
2 对mei6段增强以后的数据，汇总，打乱，预处理，而后存入 本地
  标准化处理 放在run.py中
"""
# 磨床数据预处理
import os
from scipy import io
import matplotlib.pyplot as plt
import numpy as np

# 故障部位标签 One-hot
normal_label          = [1, 0, 0]  # 正常轴承
Inner_Raceway_label   = [0, 1, 0]  # 内圈故障轴承 宽度0.6mm 深度0.02mm
Outer_Raceway_label   = [0, 0, 1]  # 外圈故障轴承 宽度0.6mm 深度0.02mm

'''为了保证在数据增强时，对所有的数据增强倍数一样，先把数据汇总以后再进行增强'''
# 信号采样频率
Fs = 25600
# 每个数据小片段长度
split_length = 2048
# 每个片段增强后的样本数
enhance_num = 300
mat_save_path = r"principle_axis_processing_data"


# =============================================== 读取matlab 文件 =======================================================
def load_mat_func(file_path):
    return io.loadmat(file_path)['spindle_z'].T


def get_data_list(folder_path, fault_list):
    data_list = []
    for condition_list in fault_list:
        for filename in condition_list:
            data_list.append(load_mat_func(folder_path + filename))
    return data_list


# normal
normal_folder_path = '/home/hust/Desktop/JiangSu/MT2/Spindle_Program/Project/MT2/normal/mat_data/'
normal_data_file_list = [['normal_unload_6000rpm-10-40-33.mat',
                          'normal_unload_9000rpm-10-52-31.mat',
                          'normal_unload_12000rpm-11-02-35.mat',
                          'normal_unload_15000rpm-11-14-23.mat'],
                         ['normal_steel304_6000rpm_feed2500_depth0.1_width3-14-45-53.mat',
                          'normal_steel304_9000rpm_feed2500_depth0.1_width3-14-53-31.mat',
                          'normal_steel304_12000rpm_feed2500_depth0.1_width3-15-07-51.mat',
                          'normal_steel304_15000rpm_feed2500_depth0.1_width3-15-18-00.mat'],
                         ['normal_al7075_6000rpm_feed2500_depth0.1_width3-9-52-48.mat',
                          'normal_al7075_9000rpm_feed2500_depth0.1_width3-10-02-46.mat',
                          'normal_al7075_12000rpm_feed2500_depth0.1_width3-10-14-36.mat',
                          'normal_al7075_15000rpm_feed2500_depth0.1_width3-10-26-58.mat']]
normal_data_list = get_data_list(normal_folder_path, normal_data_file_list)


# Inner Raceway
inner_folder_path = '/home/hust/Desktop/wzs/MT2_data/inner_0.6_0.04/mat_data/'
inner_data_file_list = [['20190422094004_inner_0.6_0.04_unload_6000rpm.mat',
                         '20190422095259_inner_0.6_0.04_unload_9000rpm.mat',
                         '20190422100513_inner_0.6_0.04_unload_12000rpm.mat',
                         '20190422101722_inner_0.6_0.04_unload_15000rpm.mat'],
                        ['20190424144412_inner_0.6_0.04_steel304_6000rpm_depth0.1_width3_feed2500.mat',
                         '20190424145121_inner_0.6_0.04_steel304_9000rpm_depth0.1_width3_feed2500.mat',
                         '20190424150254_inner_0.6_0.04_steel304_12000rpm_depth0.1_width3_feed2500.mat',
                         '20190424151432_inner_0.6_0.04_steel304_15000rpm_depth0.1_width3_feed2500.mat'],
                        ['20190423093046_inner_0.6_0.04_al7075_7000rpm_depth0.1_width3_feed2500.mat',
                         '20190423092229_inner_0.6_0.04_al7075_9000rpm_depth0.1_width3_feed2500.mat',
                         '20190423091826_inner_0.6_0.04_al7075_10000rpm_depth0.1_width3_feed2500.mat',
                         '20190423085944_inner_0.6_0.04_al7075_14000rpm_depth0.1_width3_feed2500.mat']]
Inner_Raceway_data_list = get_data_list(inner_folder_path, inner_data_file_list)


# Outer Raceway
outer_folder_path = '/home/hust/Desktop/wzs/MT2_data/outer_0.6_0.04_criticalG/mat_data/'
outer_data_file_list = [['outer-0.6-0.04_criticalG_unload_6000rpm-11-03-23.mat',
                         'outer-0.6-0.04_criticalG_unload_9000rpm-11-20-46.mat',
                         'outer-0.6-0.04_criticalG_unload_12000rpm-11-38-05.mat',
                         'outer-0.6-0.04_criticalG_unload_15000rpm-11-55-16.mat'],
                        ['outer-0.6-0.04_criticalG_steel304_6000rpm_feed2500_depth0.1_width3-14-50-21.mat',
                         'outer-0.6-0.04_criticalG_steel304_9000rpm_feed2500_depth0.1_width3-14-56-35.mat',
                         'outer-0.6-0.04_criticalG_steel304_12000rpm_feed2500_depth0.1_width3-14-37-30.mat',
                         'outer-0.6-0.04_criticalG_steel304_15000rpm_feed2500_depth0.1_width3-14-47-00.mat'],
                        ['outer-0.6-0.04_criticalG_al7075_6000rpm_feed2500_depth0.1_width3-15-10-49.mat',
                         'outer-0.6-0.04_criticalG_al7075_9000rpm_feed2500_depth0.1_width3-15-34-39.mat',
                         'outer-0.6-0.04_criticalG_al7075_12000rpm_feed2500_depth0.1_width3-15-47-22.mat',
                         'outer-0.6-0.04_criticalG_al7075_15000rpm_feed2500_depth0.1_width3-15-58-55.mat']]
Outer_Raceway_data_list = get_data_list(outer_folder_path, outer_data_file_list)


# 信号截取段
signal_split = {
    # normal
    'Normal':
        [
            # normal_unload_6000rpm   每段250000
            [[200000, 450000], [800000, 1050000], [1400000, 1650000],
             [1900000, 2150000], [2600000, 2850000], [3500000, 3750000]],
            # normal_unload_9000rpm   每段250000
            [[200000, 450000], [800000, 1050000], [1400000, 1650000],
             [1900000, 2150000], [2600000, 2850000], [3500000, 3750000]],
            # normal_unload_12000rpm   每段250000
            [[200000, 450000], [800000, 1050000], [1400000, 1650000],
             [1900000, 2150000], [2600000, 2850000], [3500000, 3750000]],
            # normal_unload_15000rpm   每段250000
            [[200000, 450000], [800000, 1050000], [1400000, 1650000],
             [1900000, 2150000], [2600000, 2850000], [3500000, 3750000]],

            # normal_steel304_6000rpm   每段60000
            [[150000, 210000], [780000, 840000], [1410000, 1470000],
             [2040000, 2100000], [2570000, 2630000], [3090000, 3150000]],

            # normal_steel304_9000rpm   每段60000
            [[305000, 365000], [830000, 890000], [1355000, 1415000],
             [1985000, 2045000], [2405000, 2465000], [3035000, 3095000]],

            # normal_steel304_12000rpm   每段60000
            [[565000, 625000], [980000, 1040000], [1405000, 1465000],
             [2040000, 2100000], [2560000, 2620000], [3300000, 3360000]],

            # normal_steel304_15000rpm   每段60000
            [[375000, 435000], [795000, 855000], [1215000, 1275000],
             [1525000, 1585000], [2160000, 2220000], [3320000, 3380000]],

            # normal_al7075_6000rpm    每段60000
            [[570000, 630000], [1120000, 1180000], [1535000, 1595000],
             [2095000, 2155000], [2920000, 2980000], [4445000, 4505000]],

            # normal_al7075_9000rpm   每段60000
            [[1320000, 1380000], [2040000, 2100000], [2505000, 2565000],
             [3120000, 3180000], [3440000, 3500000], [4620000, 4680000]],

            # normal_al7075_12000rpm    每段60000
            [[530000, 590000], [1000000, 1060000], [1520000, 1580000],
             [2200000, 2260000], [3010000, 3070000], [4480000, 4540000]],

            # normal_al7075_15000rpm   每段60000
            [[520000, 580000], [1230000, 1290000], [1900000, 1960000],
             [2720000, 2780000], [3230000, 3290000], [4785000, 4845000]],
        ],

    # Inner Raceway
    'Inner_Raceway':
        [
            # inner_0.6_0.04_unload_6000rpm  每段250000
            [[400000, 650000], [1000000, 1250000], [1600000, 1850000],
             [2100000, 2350000], [2800000, 3050000], [3700000, 3950000]],

            # inner_0.6_0.04_unload_9000rpm   每段250000
            [[400000, 650000], [1000000, 1250000], [1600000, 1850000],
             [2100000, 2350000], [2800000, 3050000], [3700000, 3950000]],

            # inner_0.6_0.04_unload_12000rpm  每段250000
            [[500000, 750000], [1100000, 1350000], [1700000, 1950000],
             [2300000, 2550000], [2900000, 3150000], [3700000, 3950000]],

            # inner_0.6_0.04_unload_15000rpm   每段250000
            [[400000, 650000], [1000000, 1250000], [1600000, 1850000],
             [2100000, 2350000], [2800000, 3050000], [3400000, 3650000]],

            # inner_0.6_0.04_steel304_6000rpm   每段250000
            [[800000, 1050000], [1300000, 1550000], [1800000, 2050000],
             [2300000, 2550000], [2800000, 3050000], [3400000, 3650000]],

            # inner_0.6_0.04_steel304_9000rpm   每段250000
            [[400000, 650000], [1000000, 1250000], [1600000, 1850000],
             [2100000, 2350000], [2600000, 2850000], [3000000, 3250000]],

            # inner_0.6_0.04_steel304_12000rpm   每段250000
            [[400000, 650000], [1000000, 1250000], [1600000, 1850000],
             [2100000, 2350000], [2600000, 2850000], [3000000, 3250000]],

            # inner_0.6_0.04_steel304_15000rpm   每段250000
            [[400000, 650000], [1000000, 1250000], [1600000, 1850000],
             [2100000, 2350000], [2600000, 2850000], [3000000, 3250000]],

            # inner_0.6_0.04_al7075_7000rpm   每段250000
            [[400000, 650000], [1000000, 1250000], [1600000, 1850000],
             [2100000, 2350000], [2800000, 3050000], [3700000, 3950000]],

            # inner_0.6_0.04_al7075_9000rpm  每段250000
            [[400000, 650000], [1000000, 1250000], [1600000, 1850000],
             [2100000, 2350000], [2800000, 3050000], [3700000, 3950000]],

            # inner_0.6_0.04_al7075_10000rpm   每段250000
            [[400000, 650000], [1000000, 1250000], [1600000, 1850000],
             [2100000, 2350000], [2800000, 3050000], [3700000, 3950000]],

            # inner_0.6_0.04_al7075_14000rpm   每段250000
            [[800000, 1050000], [1500000, 1750000], [2100000, 2350000],
             [2600000, 2850000], [3200000, 3450000], [3900000, 4150000]],
        ],

    # Outer Raceway
    "Outer_Raceway":
        [
            # outer-0.6-0.04_criticalG_unload_6000rpm   每段250000
            [[400000, 650000], [1000000, 1250000], [1600000, 1850000],
             [2100000, 2350000], [2800000, 3050000], [3700000, 3950000]],

            # outer-0.6-0.04_criticalG_unload_9000rpm   每段250000
            [[400000, 650000], [1000000, 1250000], [1600000, 1850000],
             [2100000, 2350000], [2800000, 3050000], [3700000, 3950000]],

            # outer-0.6-0.04_criticalG_unload_12000rpm    每段250000
            [[1200000, 1450000], [1800000, 2050000], [2600000, 2850000],
             [3200000, 3450000], [4000000, 4200000], [4800000, 5000000]],

            # outer-0.6-0.04_criticalG_unload_15000rpm   每段250000
            [[1200000, 1450000], [1800000, 2050000], [2600000, 2850000],
             [3200000, 3450000], [4000000, 4200000], [4800000, 5000000]],

            # outer-0.6-0.04_criticalG_steel304_6000rpm   每段60000
            [[528000, 588000], [845000, 905000], [1370000, 1430000],
             [1895000, 1955000], [2630000, 2690000], [3575000, 3635000]],

            # outer-0.6-0.04_criticalG_steel304_9000rpm   每段60000
            [[465000, 525000], [885000, 945000], [1408000, 1468000],
             [1828000, 1888000], [2565000, 2625000], [3508000, 3568000]],

            # outer-0.6-0.04_criticalG_steel304_12000rpm   每段60000
            [[380000, 440000], [695000, 755000], [1425000, 1485000],
             [1640000, 1700000], [2060000, 2120000], [3330000, 3390000]],

            # outer-0.6-0.04_criticalG_steel304_15000rpm   每段60000
            [[365000, 425000], [995000, 1055000], [1420000, 1480000],
             [1835000, 1895000], [2365000, 2425000], [3095000, 3155000]],

            # outer-0.6-0.04_criticalG_al7075_6000rpm  每段60000
            [[1200000, 1260000], [1775000, 1835000], [2635000, 2695000],
             [2920000, 2980000], [4070000, 4130000], [4795000, 4855000]],

            # outer-0.6-0.04_criticalG_al7075_9000rpm   每段60000
            [[480000, 540000], [1200000, 1260000], [1780000, 1840000],
             [2485000, 25450000], [3355000, 3415000], [4515000, 4575000]],

            # outer-0.6-0.04_criticalG_al7075_12000rpm    每段60000
            [[400000, 460000], [1015000, 1075000], [1630000, 16900000],
             [2330000, 2390000], [2820000, 2880000], [4085000, 4145000]],

            # outer-0.6-0.04_criticalG_al7075_15000rpm   每段60000
            [[300000, 360000], [810000, 870000], [1340000, 1400000],
             [2060000, 2120000], [3580000, 3640000], [4360000, 4420000]],
        ]
}


def data_enhance(array, length, multip):
    """
    数据增强，使用长度为length的窗口进行滑动，选取数据，数据之间有重叠
    使数据量变为原理的multip倍

    Parameters
    ----------
    array  : array 例如 (60000, )
    length : 截取数据的窗口长度，2048
    multip ：数据增强的倍数multiple，300

    Returns
    ----------
    array_enhance : 增强以后的数据  (300,2048)
    """
    print(" array.shape", array.shape)  # (250000, 1)
    array_enhance = np.empty([length , multip])  # (2048,43200)
    array_len = len(array) - length
    overlap = int(array_len/(multip - 1))  # 窗口滑移时的重叠长度
    print("overlap: ", overlap)  # 829
    for i in range(int(multip/2)):
        array_enhance[:, i] = array[(overlap * i) : (overlap * i + length),0]  # 从前往后写入数据
        # print("array[(overlap * i) : (overlap * i + length)].shape", array[(overlap * i) : (overlap * i + length)].shape)  # 2048
        # print("array_enhance[:, i].shape", array_enhance[:, i].shape)
        array_enhance[:, multip -i -1] = array[(array_len - overlap * i): (array_len - overlap * i + length),0]  # 从后往前写入数据
    if multip % 2 == 1:
        array_enhance[:, int(multip / 2)] = array[int(array_len / 2) : int(array_len / 2 + length),0]  # 如果multip是奇数则中间再插补一个2048的数据
    print("array_enhance.T.shape",array_enhance.T.shape)  # (300, 2048)
    return array_enhance.T


def main_signal_process(data_list, signal_split_key, one_hot_label ):
    """
    main_funtion
    对数据进行预处理

    Parameters
    ----------
    data_list  : 如：[normal_1000, normal_2000, normal_3000, normal_4000]的数据组成的列表
    signal_split_key : signal_split 数据截取字典的 key 如 "Normal"
    name_list ： 如：normal_name_list = ['normal_1000', 'normal_2000','normal_3000','normal_4000']
    length : 每个数据小片段长度
    multip : 数据增强倍数
    one_hot_label : 标签one-hot形式  如 [1,0,0]
    save_mat_name_list : 数据预处理后保存的列表，如 ['normal_data.mat', 'normal_label.mat']

    Returns
    ----------
    data_normalized  : (14400, 2048)
    label_one_hot    : (14400, 3)
    """

    per_fault_data = []   # 每一种轴承故障状态下的数据
    for index, data in enumerate(data_list):               # 对normal中12种进给速度的数据循环
        print("index", index, '\n', "data.shape:", data.shape) # 没进行切分时的原始长度，(4326400, 1)
        per_feed_data = []                                 # 某种轴承故障状态下，每一种进给速度下的数据

        for item in signal_split_key[index]: # 当index == 0，对normal中第1种速度的6个片段循环，item返回6个区间
            print("index", index)
            print("item:", item)                                           # 如 [200000, 450000]
            vibration = data[item[0]: item[1]]                           # 数据截取，如 data(200000, 450000)
            print("vibration.shape", vibration.shape)                      # 每一小段数据的长度，(250000, 1)
            enhanced_vibration = data_enhance(vibration, split_length, enhance_num)  # 数据增强
            print("enhanced vibration data.shape", enhanced_vibration.shape)   # (300, 2048)

            per_feed_data.append(enhanced_vibration)                          #

        print("per_feed_data.shape", np.array(per_feed_data).shape)         # 纵向堆叠，每种转速下的6段数据(6, 300, 2048)
        per_feed_data_array = np.concatenate(per_feed_data,axis=0)
        print("per_feed_data_array.shape", np.array(per_feed_data_array).shape)        # (1800, 2048)
        per_fault_data.append(per_feed_data_array)

    print("per_fault_data.shape",np.array(per_fault_data).shape)         # (12, 1800, 2048) 所有转速种类堆叠，12种转速
    per_fault_data_array = np.concatenate(per_fault_data, axis=0)
    print("per_fault_data_array.shape",per_fault_data_array.shape)       # (21600, 2048)

    label         = np.array(one_hot_label * len(per_fault_data_array))
    label_one_hot = label.reshape(len(per_fault_data_array), -1)
    print("label_one_hot.shape",label_one_hot.shape)                     # (21600, 3)

    num_sample_per_fault = len(per_fault_data_array)
    index_permutation = np.arange(num_sample_per_fault)
    np.random.shuffle(index_permutation)
    shuffled_data = per_fault_data_array[index_permutation][          : num_sample_per_fault]
    print("shuffled_data.shape",shuffled_data.shape)                     # (21600, 2048)

    return shuffled_data, label_one_hot


'''================================================== 主函数调用 ============================================'''
#  数据预处理
print('='*100)
# Normal
data_normal,        label_normal        = main_signal_process(data_list=normal_data_list,                 # 12 个文件的数据列表
                                                signal_split_key = signal_split['Normal'],    # 12个 normal轴承的6个片段（12*6）
                                                one_hot_label = normal_label )                # norm_label = [1, 0, 0]

# Inner_Raceway
data_Inner_Raceway, label_Inner_Raceway = main_signal_process(data_list=Inner_Raceway_data_list,  # 12 个文件的数据列表
                                             signal_split_key=signal_split['Inner_Raceway'],      # 12 个 normal 轴承的6个片段（12*6）
                                             one_hot_label=Inner_Raceway_label )                  # Inner_Raceway_label = [0, 1, 0]

# Outer_Raceway
data_Outer_Raceway, label_Outer_Raceway = main_signal_process(data_list=Outer_Raceway_data_list,  # 12个文件的数据列表
                                             signal_split_key=signal_split['Outer_Raceway'],      # 12个 normal轴承的6个片段（12*6）
                                             one_hot_label=Outer_Raceway_label )                  # Outer_Raceway_label = [0, 0, 1]

# 保存文件
save_mat_name = 'fault_bearing_dataset.mat'
data_save_path  = os.path.join(mat_save_path, save_mat_name)

fault_bearing_dataset  = {'normal_data'         : data_normal,
                          'normal_label'        : label_normal,
                          'Inner_Raceway_data'  : data_Inner_Raceway,
                          'Inner_Raceway_label' : label_Inner_Raceway,
                          'Outer_Raceway_data'  : data_Outer_Raceway,
                          'Outer_Raceway_label' : label_Outer_Raceway}     # 以字典形式保存为 mat 文件

io.savemat (data_save_path, fault_bearing_dataset)

print("============================================  All finished  =======================================")




'''
/home/hust/anaconda3/envs/mypython/bin/python /home/hust/Desktop/wzs/principle_axis/data_processing.py
====================================================================================================
index 0 
 data.shape: (4326400, 1)
index 0
item: [200000, 450000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 0
item: [800000, 1050000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 0
item: [1400000, 1650000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 0
item: [1900000, 2150000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 0
item: [2600000, 2850000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 0
item: [3500000, 3750000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 1 
 data.shape: (4096000, 1)
index 1
item: [200000, 450000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 1
item: [800000, 1050000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 1
item: [1400000, 1650000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 1
item: [1900000, 2150000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 1
item: [2600000, 2850000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 1
item: [3500000, 3750000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 2 
 data.shape: (5273600, 1)
index 2
item: [200000, 450000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 2
item: [800000, 1050000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 2
item: [1400000, 1650000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 2
item: [1900000, 2150000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 2
item: [2600000, 2850000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 2
item: [3500000, 3750000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 3 
 data.shape: (4556800, 1)
index 3
item: [200000, 450000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 3
item: [800000, 1050000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 3
item: [1400000, 1650000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 3
item: [1900000, 2150000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 3
item: [2600000, 2850000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 3
item: [3500000, 3750000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 4 
 data.shape: (3916800, 1)
index 4
item: [150000, 210000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 4
item: [780000, 840000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 4
item: [1410000, 1470000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 4
item: [2040000, 2100000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 4
item: [2570000, 2630000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 4
item: [3090000, 3150000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 5 
 data.shape: (3840000, 1)
index 5
item: [305000, 365000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 5
item: [830000, 890000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 5
item: [1355000, 1415000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 5
item: [1985000, 2045000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 5
item: [2405000, 2465000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 5
item: [3035000, 3095000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 6 
 data.shape: (4147200, 1)
index 6
item: [565000, 625000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 6
item: [980000, 1040000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 6
item: [1405000, 1465000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 6
item: [2040000, 2100000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 6
item: [2560000, 2620000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 6
item: [3300000, 3360000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 7 
 data.shape: (4070400, 1)
index 7
item: [375000, 435000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 7
item: [795000, 855000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 7
item: [1215000, 1275000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 7
item: [1525000, 1585000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 7
item: [2160000, 2220000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 7
item: [3320000, 3380000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 8 
 data.shape: (5222400, 1)
index 8
item: [570000, 630000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 8
item: [1120000, 1180000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 8
item: [1535000, 1595000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 8
item: [2095000, 2155000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 8
item: [2920000, 2980000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 8
item: [4445000, 4505000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 9 
 data.shape: (5606400, 1)
index 9
item: [1320000, 1380000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 9
item: [2040000, 2100000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 9
item: [2505000, 2565000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 9
item: [3120000, 3180000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 9
item: [3440000, 3500000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 9
item: [4620000, 4680000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 10 
 data.shape: (5248000, 1)
index 10
item: [530000, 590000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 10
item: [1000000, 1060000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 10
item: [1520000, 1580000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 10
item: [2200000, 2260000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 10
item: [3010000, 3070000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 10
item: [4480000, 4540000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 11 
 data.shape: (5299200, 1)
index 11
item: [520000, 580000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 11
item: [1230000, 1290000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 11
item: [1900000, 1960000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 11
item: [2720000, 2780000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 11
item: [3230000, 3290000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 11
item: [4785000, 4845000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
per_fault_data.shape (12, 1800, 2048)
per_fault_data_array.shape (21600, 2048)
label_one_hot.shape (21600, 3)
shuffled_data.shape (21600, 2048)
index 0 
 data.shape: (4733440, 1)
index 0
item: [400000, 650000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 0
item: [1000000, 1250000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 0
item: [1600000, 1850000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 0
item: [2100000, 2350000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 0
item: [2800000, 3050000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 0
item: [3700000, 3950000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 1 
 data.shape: (4456960, 1)
index 1
item: [400000, 650000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 1
item: [1000000, 1250000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 1
item: [1600000, 1850000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 1
item: [2100000, 2350000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 1
item: [2800000, 3050000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 1
item: [3700000, 3950000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 2 
 data.shape: (5230080, 1)
index 2
item: [500000, 750000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 2
item: [1100000, 1350000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 2
item: [1700000, 1950000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 2
item: [2300000, 2550000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 2
item: [2900000, 3150000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 2
item: [3700000, 3950000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 3 
 data.shape: (4293120, 1)
index 3
item: [400000, 650000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 3
item: [1000000, 1250000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 3
item: [1600000, 1850000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 3
item: [2100000, 2350000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 3
item: [2800000, 3050000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 3
item: [3400000, 3650000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 4 
 data.shape: (4055040, 1)
index 4
item: [800000, 1050000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 4
item: [1300000, 1550000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 4
item: [1800000, 2050000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 4
item: [2300000, 2550000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 4
item: [2800000, 3050000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 4
item: [3400000, 3650000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 5 
 data.shape: (3850240, 1)
index 5
item: [400000, 650000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 5
item: [1000000, 1250000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 5
item: [1600000, 1850000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 5
item: [2100000, 2350000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 5
item: [2600000, 2850000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 5
item: [3000000, 3250000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 6 
 data.shape: (3486720, 1)
index 6
item: [400000, 650000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 6
item: [1000000, 1250000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 6
item: [1600000, 1850000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 6
item: [2100000, 2350000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 6
item: [2600000, 2850000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 6
item: [3000000, 3250000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 7 
 data.shape: (3665920, 1)
index 7
item: [400000, 650000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 7
item: [1000000, 1250000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 7
item: [1600000, 1850000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 7
item: [2100000, 2350000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 7
item: [2600000, 2850000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 7
item: [3000000, 3250000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 8 
 data.shape: (4648960, 1)
index 8
item: [400000, 650000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 8
item: [1000000, 1250000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 8
item: [1600000, 1850000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 8
item: [2100000, 2350000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 8
item: [2800000, 3050000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 8
item: [3700000, 3950000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 9 
 data.shape: (4718080, 1)
index 9
item: [400000, 650000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 9
item: [1000000, 1250000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 9
item: [1600000, 1850000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 9
item: [2100000, 2350000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 9
item: [2800000, 3050000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 9
item: [3700000, 3950000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 10 
 data.shape: (4702720, 1)
index 10
item: [400000, 650000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 10
item: [1000000, 1250000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 10
item: [1600000, 1850000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 10
item: [2100000, 2350000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 10
item: [2800000, 3050000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 10
item: [3700000, 3950000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 11 
 data.shape: (5614080, 1)
index 11
item: [800000, 1050000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 11
item: [1500000, 1750000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 11
item: [2100000, 2350000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 11
item: [2600000, 2850000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 11
item: [3200000, 3450000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 11
item: [3900000, 4150000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
per_fault_data.shape (12, 1800, 2048)
per_fault_data_array.shape (21600, 2048)
label_one_hot.shape (21600, 3)
shuffled_data.shape (21600, 2048)
index 0 
 data.shape: (6144000, 1)
index 0
item: [400000, 650000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 0
item: [1000000, 1250000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 0
item: [1600000, 1850000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 0
item: [2100000, 2350000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 0
item: [2800000, 3050000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 0
item: [3700000, 3950000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 1 
 data.shape: (7552000, 1)
index 1
item: [400000, 650000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 1
item: [1000000, 1250000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 1
item: [1600000, 1850000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 1
item: [2100000, 2350000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 1
item: [2800000, 3050000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 1
item: [3700000, 3950000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 2 
 data.shape: (7603200, 1)
index 2
item: [1200000, 1450000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 2
item: [1800000, 2050000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 2
item: [2600000, 2850000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 2
item: [3200000, 3450000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 2
item: [4000000, 4200000]
vibration.shape (200000, 1)
 array.shape (200000, 1)
overlap:  662
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 2
item: [4800000, 5000000]
vibration.shape (200000, 1)
 array.shape (200000, 1)
overlap:  662
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 3 
 data.shape: (7705600, 1)
index 3
item: [1200000, 1450000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 3
item: [1800000, 2050000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 3
item: [2600000, 2850000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 3
item: [3200000, 3450000]
vibration.shape (250000, 1)
 array.shape (250000, 1)
overlap:  829
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 3
item: [4000000, 4200000]
vibration.shape (200000, 1)
 array.shape (200000, 1)
overlap:  662
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 3
item: [4800000, 5000000]
vibration.shape (200000, 1)
 array.shape (200000, 1)
overlap:  662
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 4 
 data.shape: (4249600, 1)
index 4
item: [528000, 588000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 4
item: [845000, 905000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 4
item: [1370000, 1430000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 4
item: [1895000, 1955000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 4
item: [2630000, 2690000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 4
item: [3575000, 3635000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 5 
 data.shape: (4121600, 1)
index 5
item: [465000, 525000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 5
item: [885000, 945000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 5
item: [1408000, 1468000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 5
item: [1828000, 1888000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 5
item: [2565000, 2625000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 5
item: [3508000, 3568000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 6 
 data.shape: (4172800, 1)
index 6
item: [380000, 440000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 6
item: [695000, 755000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 6
item: [1425000, 1485000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 6
item: [1640000, 1700000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 6
item: [2060000, 2120000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 6
item: [3330000, 3390000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 7 
 data.shape: (4121600, 1)
index 7
item: [365000, 425000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 7
item: [995000, 1055000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 7
item: [1420000, 1480000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 7
item: [1835000, 1895000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 7
item: [2365000, 2425000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 7
item: [3095000, 3155000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 8 
 data.shape: (6016000, 1)
index 8
item: [1200000, 1260000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 8
item: [1775000, 1835000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 8
item: [2635000, 2695000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 8
item: [2920000, 2980000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 8
item: [4070000, 4130000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 8
item: [4795000, 4855000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 9 
 data.shape: (5376000, 1)
index 9
item: [480000, 540000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 9
item: [1200000, 1260000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 9
item: [1780000, 1840000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 9
item: [2485000, 25450000]
vibration.shape (2891000, 1)
 array.shape (2891000, 1)
overlap:  9662
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 9
item: [3355000, 3415000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 9
item: [4515000, 4575000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 10 
 data.shape: (5248000, 1)
index 10
item: [400000, 460000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 10
item: [1015000, 1075000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 10
item: [1630000, 16900000]
vibration.shape (3618000, 1)
 array.shape (3618000, 1)
overlap:  12093
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 10
item: [2330000, 2390000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 10
item: [2820000, 2880000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 10
item: [4085000, 4145000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
index 11 
 data.shape: (5145600, 1)
index 11
item: [300000, 360000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 11
item: [810000, 870000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 11
item: [1340000, 1400000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 11
item: [2060000, 2120000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 11
item: [3580000, 3640000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
index 11
item: [4360000, 4420000]
vibration.shape (60000, 1)
 array.shape (60000, 1)
overlap:  193
array_enhance.T.shape (300, 2048)
enhanced vibration data.shape (300, 2048)
per_feed_data.shape (6, 300, 2048)
per_feed_data_array.shape (1800, 2048)
per_fault_data.shape (12, 1800, 2048)
per_fault_data_array.shape (21600, 2048)
label_one_hot.shape (21600, 3)
shuffled_data.shape (21600, 2048)
============================================  All finished  =======================================

Process finished with exit code 0
'''







