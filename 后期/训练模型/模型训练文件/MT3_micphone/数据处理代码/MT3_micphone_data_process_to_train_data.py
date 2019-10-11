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
mat_save_path = r"micphone_MT3_processing_data"


# =============================================== 读取matlab 文件 =======================================================
def load_mat_func(file_path):
    return io.loadmat(file_path)['micphone_data'].T


def get_data_list(folder_path, fault_list):
    data_list = []
    for condition_list in fault_list:
        for filename in condition_list:
            data_list.append(load_mat_func(folder_path + filename))
    return data_list


# normal
normal_folder_path = '/home/hust/Desktop/wzs/micphone_analysis/micphone_mat/normal/'
normal_data_file_list = [['x_normal_feed2000-14-28-50.mat',
                          'x_normal_feed2500-14-29-26.mat',
                          'x_normal_feed3000-14-30-00.mat',
                          'x_normal_feed4000-14-30-30.mat'],  # small
                         ['new_all_feed_x_10000-9-29-06.mat',
                          'x_normal_feed10000-14-33-29.mat',
                          'new_all_feed_x_20000-9-31-21.mat',
                          'x_normal_feed20000-14-34-09.mat']]  # large
normal_data_list = get_data_list(normal_folder_path, normal_data_file_list)


# Inner Raceway
inner_folder_path = '/home/hust/Desktop/wzs/micphone_analysis/micphone_mat/inner/'
inner_data_file_list = [['x_in_0.6_0.02_bear2_feed2000-19-57-11.mat',
                         'x_in_0.6_0.02_bear2_feed2500-19-57-52.mat',
                         'x_in_0.6_0.02_bear2_feed3000-19-58-16.mat',
                         'x_in_0.6_0.02_bear2_feed4000-19-58-35.mat'],  # small
                        ['x_0.6_0.02_num4_feed10000-19-03-27.mat',
                         'x_0.6_0.02_num5_feed10000-15-08-05.mat',
                         'x_0.6_0.02_num4_feed20000-19-04-34.mat',
                         'x_0.6_0.02_num5_feed20000-15-05-30.mat']]  # large
Inner_Raceway_data_list = get_data_list(inner_folder_path, inner_data_file_list)


# Outer Raceway
outer_folder_path = '/home/hust/Desktop/wzs/micphone_analysis/micphone_mat/outer/'
outer_data_file_list = [['x_out_0.6_0.02_bear4_feed2000-15-36-47.mat',
                         'x_out_0.6_0.02_bear4_feed2500-15-37-14.mat',
                         'x_out_0.6_0.02_bear4_feed3000-15-37-39.mat',
                         'x_out_0.6_0.02_bear4_feed4000-15-38-17.mat'],  # small
                        ['x_out_0.6_0.02_num6_feed10000-15-07-23.mat',
                         'x_out_deg180_0.6_0.02_num6_feed10000-19-53-33.mat',
                         'x_out_0.6_0.02_num6_feed20000-15-06-31.mat',
                         'x_out_deg180_0.6_0.02_num6_feed20000-19-52-59.mat']]  # large
Outer_Raceway_data_list = get_data_list(outer_folder_path, outer_data_file_list)


# 信号截取段
signal_split = {
    # normal
    'Normal':
        [
            # x_normal_feed2000-14-28-50.tdms          1个来回  每段长度60000
            [[85000, 145000], [145000, 205000], [205000, 265000], [265000, 325000], [325000, 385000], [385000, 445000]],

            # x_normal_feed2500-14-29-26.tdms          1个来回  每段长度45000
            [[110000, 155000], [155000, 200000], [200000, 245000], [245000, 290000], [290000, 335000], [335000, 380000]],

            # x_normal_feed3000-14-30-00.tdms          1个来回  每段 35000
            [[90000, 125000], [125000, 160000], [160000, 195000], [195000, 230000], [230000, 265000], [265000, 300000]],

            # x_normal_feed4000-14-30-30.tdms           3个来回  每段80000
            [[120000, 200000], [210000, 290000], [300000, 380000], [390000, 470000], [480000, 560000], [580000, 660000]],

            # new_all_feed_x_10000-9-29-06.tdms    每段间隔60000个点
            [[235000, 295000], [305000, 365000], [385000, 445000], [455000, 515000], [535000, 595000], [605000, 665000]],

            # x_normal_feed10000-14-33-29.mat   每段间隔25000个点
            [[110000, 135000], [155000, 180000], [190000, 215000], [225000, 250000], [265000, 290000], [330000, 355000]],

            # new_all_feed_x_20000-9-31-21.tdms    每段间隔30000个点
            [[60000, 90000], [95000, 125000], [145000, 165000], [173000, 203000], [210000, 240000], [250000, 280000]],

            # x_normal_feed20000-14-34-09.mat    每段间隔25000个点
            [[75000, 100000], [100000, 125000], [125000, 158000], [160000, 185000], [195000, 220000], [225000, 250000]],
        ],

    # Inner Raceway
    'Inner_Raceway':
        [
            # x_in_0.6_0.02_bear2_feed2000-19-57-11.tdms  1个来回 每段长度60000
            [[60000, 120000], [120000, 180000], [180000, 240000], [240000, 300000], [300000, 360000], [360000, 400000]],

            # x_in_0.6_0.02_bear2_feed2500-19-57-52.tdms  1个来回  每段长度45000
            [[110000, 155000], [155000, 200000], [200000, 245000], [245000, 290000], [290000, 335000], [335000, 380000]],

            # x_in_0.6_0.02_bear2_feed3000-19-58-16.tdms  1个来回  每段长度40000
            [[80000, 120000], [120000, 160000], [160000, 200000], [200000, 240000], [240000, 280000], [280000, 320000]],

            # x_in_0.6_0.02_bear2_feed4000-19-58-35.tdms   3个来回  每段长度80000
            [[69000, 149000], [160000, 240000], [260000, 340000], [350000, 430000], [440000, 520000], [530000, 610000]],

            # x_0.6_0.02_num4_feed10000-19-03-27.tdms
            [[30000, 80000], [90000, 140000], [150000, 200000], [215000, 265000], [270000, 320000], [340000, 390000]],

            # x_0.6_0.02_num5_feed10000-15-08-05.tdms
            [[150000, 200000], [210000, 260000], [270000, 320000], [335000, 385000], [395000, 445000], [455000, 505000]],

            # x_0.6_0.02_num4_feed20000-19-04-34.tdms
            [[38000, 63000], [70000, 95000], [100000, 125000], [130000, 155000], [160000, 185000], [192000, 217000]],

            # x_0.6_0.02_num5_feed20000-15-05-30.tdms
            [[75000, 100000], [105000, 130000], [135000, 160000], [165000, 190000], [197000, 222000], [228000, 253000]],
        ],

    # Outer Raceway
    "Outer_Raceway":
        [
            # x_out_0.6_0.02_bear4_feed2000-15-36-47.tdms  1个来回  每段长度55000
            [[25000, 80000], [80000, 135000], [135000, 190000], [190000, 245000], [245000, 300000], [300000, 355000]],

            # x_out_0.6_0.02_bear4_feed2500-15-37-14.tdms  1个来回  每段长度45000
            [[40000, 85000], [85000, 130000], [130000, 175000], [175000, 220000], [220000, 265000], [265000, 310000]],

            # x_out_0.6_0.02_bear4_feed3000-15-37-39.tdms  1个来回  每段长度40000
            [[110000, 150000], [150000, 190000], [190000, 230000], [230000, 270000], [270000, 310000], [310000, 350000]],

            # x_out_0.6_0.02_bear4_feed4000-15-38-17.tdms  3个来回  每段长度80000
            [[47000, 127000], [140000, 220000], [230000, 310000], [325000, 405000], [410000, 490000], [510000, 590000]],

            [[70000, 120000], [130000, 180000], [190000, 240000], [250000, 300000], [315000, 365000], [380000, 430000]],

            # x_out_deg180_0.6_0.02_num6_feed10000-19-53-33.tdms
            [[40000, 90000], [105000, 155000], [165000, 215000], [225000, 275000], [290000, 340000], [350000, 400000]],

            # x_out_0.6_0.02_num6_feed20000-15-06-31.tdms
            [[60000, 85000], [90000, 115000], [123000, 148000], [152000, 177000], [182000, 207000], [215000, 240000]],

            # x_out_deg180_0.6_0.02_num6_feed20000-19-52-59.tdms
            [[32000, 57000], [63000, 88000], [95000, 120000], [125000, 150000], [155000, 180000], [187000, 204799]],
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
save_mat_name = 'micphone_dataset.mat'
data_save_path  = os.path.join(mat_save_path, save_mat_name)

fault_bearing_dataset  = {'normal_data'         : data_normal,
                          'normal_label'        : label_normal,
                          'Inner_Raceway_data'  : data_Inner_Raceway,
                          'Inner_Raceway_label' : label_Inner_Raceway,
                          'Outer_Raceway_data'  : data_Outer_Raceway,
                          'Outer_Raceway_label' : label_Outer_Raceway}     # 以字典形式保存为 mat 文件

io.savemat (data_save_path, fault_bearing_dataset)

print("============================================  All finished  =======================================")
