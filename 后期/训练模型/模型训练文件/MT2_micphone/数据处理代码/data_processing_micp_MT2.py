
"""【数据预处理思路】
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
mat_save_path = r"MT2_micphone_dataset"


# =============================================== 读取matlab 文件 =======================================================
def load_mat_func(file_path):
    return io.loadmat(file_path)['MT2_mic'].T


def get_data_list(folder_path, fault_list):
    data_list = []
    for condition_list in fault_list:
        for filename in condition_list:
            data_list.append(load_mat_func(folder_path + filename))
    return data_list


# normal
normal_folder_path = '/home/hust/Desktop/wzs/MT2_micphone/MT2_micphone_data/normal_slightG/'
normal_data_file_list = [['noamal_slightG_unload_6000rpm-15-48-44.mat',
                          'noamal_slightG_unload_9000rpm-16-05-17.mat',
                          'noamal_slightG_unload_12000rpm-16-20-29.mat',
                          'noamal_slightG_unload_15000rpm-16-36-04.mat'],
                         ['noamal_slightG_al7075_6000rpm_feed2500_depth0.1_width3-9-10-19.mat',
                          'noamal_slightG_al7075_9000rpm_feed2500_depth0.1_width3-9-19-53.mat',
                          'noamal_slightG_al7075_12000rpm_feed2500_depth0.1_width3-9-32-23.mat',
                          'noamal_slightG_al7075_15000rpm_feed2500_depth0.1_width3-9-45-48.mat'],
                         ['noamal_slightG_steel304_6000rpm_feed2500_depth0.1_width3-16-45-14.mat',
                          'noamal_slightG_steel304_9000rpm_feed2500_depth0.1_width3-16-51-27.mat',
                          'noamal_slightG_steel304_12000rpm_feed2500_depth0.1_width3-17-03-05.mat',
                          'noamal_slightG_steel304_15000rpm_feed2500_depth0.1_width3-17-12-18.mat']]
normal_data_list = get_data_list(normal_folder_path, normal_data_file_list)


# Inner Raceway
inner_folder_path = '/home/hust/Desktop/wzs/MT2_micphone/MT2_micphone_data/inner_0.6_0.04/'
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
outer_folder_path = '/home/hust/Desktop/wzs/MT2_micphone/MT2_micphone_data/outer-0.6-0.04_criticalG/'
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


def become_6_times(start,end):
    times = (end-start)//6
    changed_end = start+times*6
    return start, changed_end


def get_split(split_array):
    all_list_split=[]
    for (start, end) in split_array:
        start, end = become_6_times(start,end)
        list_index = np.array(range(start,end)).reshape(6,-1)
        list_split = [list((i[0],i[-1])) for i in list_index]
        all_list_split.append(list_split)
    return all_list_split


# 信号截取段
signal_split = {
    # normal
    'Normal':
            get_split([[2000000, 6500000],
                       [500000, 5000000],
                       [1000000, 6000000],
                       [2000000, 6500000],
                       [1500000, 5500000],
                       [1000000, 4000000],
                       [2000000, 4000000],
                       [1000000, 5000000],
                       [500000, 3500000],
                       [500000, 3500000],
                       [500000, 3500000],
                       [500000, 3000000]])
        ,

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
