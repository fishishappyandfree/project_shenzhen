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
mat_save_path = r"MT1_x_feed_processing_data"


# =============================================== 读取matlab 文件 =======================================================
def load_mat_func(file_path):
    return io.loadmat(file_path)['MT1_x_feed_axis_data'].T


def get_data_list(folder_path, fault_list):
    data_list = []
    for filename in fault_list:
        data_list.append(load_mat_func(folder_path + filename))
    return data_list


# normal
normal_folder_path = '/home/hust/Desktop/wzs/MT1/MT1_X_feed/x_feed_axis_mat/'
normal_data_file_list = ['MT1_x_normal_f_2000-9-36-26',  # 16000, 877500
                         'MT1_x_normal_f_3000-9-38-22',  # 65000, 545000
                         'MT1_x_normal_f_4000-9-38-59',  # 60000, 1140000
                         'MT1_x_normal_f_5000-9-39-55',  # 65000, 930000
                         'MT1_x_normal_f_6000-9-40-45',  # 55000, 775000
                         'MT1_x_normal_f_7000-9-41-26',  # 55000, 670000
                         'MT1_x_normal_f_8000-9-42-02',  # 90000, 630000
                         'MT1_x_normal_f_9000-9-42-41',  # 70000, 640000
                         'MT1_x_normal_f_10000-9-43-15']  # 160000, 730000
normal_data_list = get_data_list(normal_folder_path, normal_data_file_list)


# Inner Raceway
inner_folder_path = '/home/hust/Desktop/wzs/MT1/MT1_X_feed/x_feed_axis_mat/'
inner_data_file_list = ['MT1_x_in_0.6_0.002_f_2000-13-25-54',  # 80000, 610000
                        'MT1_x_in_0.6_0.002_f_3000-13-26-54',  # 67000, 420000
                        'MT1_x_in_0.6_0.002_f_4000-13-27-19',  # 60000, 965000
                        'MT1_x_in_0.6_0.002_f_5000-13-28-01',  # 30000, 675000
                        'MT1_x_in_0.6_0.002_f_6000-13-28-49',  # 35000, 570000
                        'MT1_x_in_0.6_0.002_f_7000-13-29-27',  # 50000, 510000
                        'MT1_x_in_0.6_0.002_f_8000-13-29-57',  # 65000, 465000
                        'MT1_x_in_0.6_0.002_f_9000-13-30-43',  # 45000, 380000
                        'MT1_x_in_0.6_0.002_f_10000-13-31-03']  # 75000, 500000
Inner_Raceway_data_list = get_data_list(inner_folder_path, inner_data_file_list)


# Outer Raceway
outer_folder_path = '/home/hust/Desktop/wzs/MT1/MT1_X_feed/x_feed_axis_mat/'
outer_data_file_list = ['MT1_x_out_0.6_0.002_f_2000-8-24-06',  # 100000, 630000
                        'MT1_x_out_0.6_0.002_f_3000-8-25-04',  # 45000, 390000
                        'MT1_x_out_0.6_0.002_f_4000-8-25-32',  # 45000, 850000
                        'MT1_x_out_0.6_0.002_f_5000-8-26-12',  # 45000, 690000
                        'MT1_x_out_0.6_0.002_f_6000-8-26-45',  # 50000, 585000
                        'MT1_x_out_0.6_0.002_f_7000-8-27-14',  # 40000, 500000
                        'MT1_x_out_0.6_0.002_f_8000-8-28-01',  # 25000, 425000
                        'MT1_x_out_0.6_0.002_f_9000-8-28-24',  # 40000, 395000
                        'MT1_x_out_0.6_0.002_f_10000-8-28-55']  # 40000, 470000
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
            get_split([[16000, 877500],
                       [65000, 545000],
                       [60000, 1140000],
                       [65000, 930000],
                       [55000, 775000],
                       [55000, 670000],
                       [90000, 630000],
                       [70000, 640000],
                       [160000, 730000]])
        ,

    # Inner Raceway
    'Inner_Raceway':
            get_split([[80000, 610000],
                       [67000, 420000],
                       [60000, 965000],
                       [30000, 675000],
                       [35000, 570000],
                       [50000, 510000],
                       [65000, 465000],
                       [45000, 380000],
                       [75000, 500000]])
        ,

    # Outer Raceway
    "Outer_Raceway":

            get_split([[100000, 630000],
                       [45000, 390000],
                       [45000, 850000],
                       [45000, 690000],
                       [50000, 585000],
                       [40000, 500000],
                       [25000, 425000],
                       [40000, 395000],
                       [40000, 470000]])

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
save_mat_name = 'MT1_x_feed_dataset.mat'
data_save_path  = os.path.join(mat_save_path, save_mat_name)

fault_bearing_dataset  = {'normal_data'         : data_normal,
                          'normal_label'        : label_normal,
                          'Inner_Raceway_data'  : data_Inner_Raceway,
                          'Inner_Raceway_label' : label_Inner_Raceway,
                          'Outer_Raceway_data'  : data_Outer_Raceway,
                          'Outer_Raceway_label' : label_Outer_Raceway}     # 以字典形式保存为 mat 文件

io.savemat (data_save_path, fault_bearing_dataset)

print("============================================  All finished  =======================================")
