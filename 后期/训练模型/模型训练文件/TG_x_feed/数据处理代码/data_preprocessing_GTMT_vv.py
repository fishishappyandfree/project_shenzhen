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


mat_save_path = r"GTMT_preprocessing_data_vv"

#=============================================== 读取matlab 文件 =======================================================
# normal
dict_normal_1000 = u'F:/_project_bigdata_from_HuTuKang/_Model\Model_for_GTMT/data_selected_1000_to_4000/normal_1000.mat'
dict_normal_2000 = u'F:/_project_bigdata_from_HuTuKang/_Model\Model_for_GTMT/data_selected_1000_to_4000/normal_2000.mat'
dict_normal_3000 = u'F:/_project_bigdata_from_HuTuKang/_Model\Model_for_GTMT/data_selected_1000_to_4000/normal_3000.mat'
dict_normal_4000 = u'F:/_project_bigdata_from_HuTuKang/_Model\Model_for_GTMT/data_selected_1000_to_4000/normal_4000.mat'
dict_normal_5000 = u'F:/_project_bigdata_from_HuTuKang/_Model\Model_for_GTMT_v2.0/data_selected_5000_to_8000/normal_5000.mat'
dict_normal_6000 = u'F:/_project_bigdata_from_HuTuKang/_Model\Model_for_GTMT_v2.0/data_selected_5000_to_8000/normal_6000.mat'
dict_normal_7000 = u'F:/_project_bigdata_from_HuTuKang/_Model\Model_for_GTMT_v2.0/data_selected_5000_to_8000/normal_7000.mat'
dict_normal_8000 = u'F:/_project_bigdata_from_HuTuKang/_Model\Model_for_GTMT_v2.0/data_selected_5000_to_8000/normal_8000.mat'
normal_1000 = io.loadmat(dict_normal_1000)['signal_save'].T   # 取出dict中的‘signal_save’数据
normal_2000 = io.loadmat(dict_normal_2000)['signal_save'].T
normal_3000 = io.loadmat(dict_normal_3000)['signal_save'].T
normal_4000 = io.loadmat(dict_normal_4000)['signal_save'].T
normal_5000 = io.loadmat(dict_normal_5000)['signal_save'].T   # 取出dict中的‘signal_save’数据
normal_6000 = io.loadmat(dict_normal_6000)['signal_save'].T
normal_7000 = io.loadmat(dict_normal_7000)['signal_save'].T
normal_8000 = io.loadmat(dict_normal_8000)['signal_save'].T

#print("normal_1000.shape",type(normal_1000),normal_1000.shape)
normal_data_list = [normal_1000, normal_2000, normal_3000, normal_4000,normal_5000, normal_6000, normal_7000, normal_8000]
# for index, data in enumerate(normal_data_list): # 对normal中4中进给速度的数据循环
#     print("index,data.shape:",index,data.shape)

# Inner Raceway
dict_Inner_Raceway_1000 = u'F:/_project_bigdata_from_HuTuKang/_Model\Model_for_GTMT/data_selected_1000_to_4000/Inner_Raceway_1000.mat'
dict_Inner_Raceway_2000 = u'F:/_project_bigdata_from_HuTuKang/_Model\Model_for_GTMT/data_selected_1000_to_4000/Inner_Raceway_2000.mat'
dict_Inner_Raceway_3000 = u'F:/_project_bigdata_from_HuTuKang/_Model\Model_for_GTMT/data_selected_1000_to_4000/Inner_Raceway_3000.mat'
dict_Inner_Raceway_4000 = u'F:/_project_bigdata_from_HuTuKang/_Model\Model_for_GTMT/data_selected_1000_to_4000/Inner_Raceway_4000.mat'
dict_Inner_Raceway_5000 = u'F:/_project_bigdata_from_HuTuKang/_Model\Model_for_GTMT_v2.0/data_selected_5000_to_8000/Inner_Raceway_5000.mat'
dict_Inner_Raceway_6000 = u'F:/_project_bigdata_from_HuTuKang/_Model\Model_for_GTMT_v2.0/data_selected_5000_to_8000/Inner_Raceway_6000.mat'
dict_Inner_Raceway_7000 = u'F:/_project_bigdata_from_HuTuKang/_Model\Model_for_GTMT_v2.0/data_selected_5000_to_8000/Inner_Raceway_7000.mat'
dict_Inner_Raceway_8000 = u'F:/_project_bigdata_from_HuTuKang/_Model\Model_for_GTMT_v2.0/data_selected_5000_to_8000/Inner_Raceway_8000.mat'
Inner_Raceway_1000 = io.loadmat(dict_Inner_Raceway_1000)['signal_save'].T
Inner_Raceway_2000 = io.loadmat(dict_Inner_Raceway_2000)['signal_save'].T
Inner_Raceway_3000 = io.loadmat(dict_Inner_Raceway_3000)['signal_save'].T
Inner_Raceway_4000 = io.loadmat(dict_Inner_Raceway_4000)['signal_save'].T
Inner_Raceway_5000 = io.loadmat(dict_Inner_Raceway_5000)['signal_save'].T
Inner_Raceway_6000 = io.loadmat(dict_Inner_Raceway_6000)['signal_save'].T
Inner_Raceway_7000 = io.loadmat(dict_Inner_Raceway_7000)['signal_save'].T
Inner_Raceway_8000 = io.loadmat(dict_Inner_Raceway_8000)['signal_save'].T
Inner_Raceway_data_list = [Inner_Raceway_1000, Inner_Raceway_2000, Inner_Raceway_3000, Inner_Raceway_4000,Inner_Raceway_5000, Inner_Raceway_6000, Inner_Raceway_7000, Inner_Raceway_8000]
# Outer Raceway
dict_Outer_Raceway_1000 = u'F:/_project_bigdata_from_HuTuKang/_Model\Model_for_GTMT/data_selected_1000_to_4000/Outer_Raceway_1000.mat'
dict_Outer_Raceway_2000 = u'F:/_project_bigdata_from_HuTuKang/_Model\Model_for_GTMT/data_selected_1000_to_4000/Outer_Raceway_2000.mat'
dict_Outer_Raceway_3000 = u'F:/_project_bigdata_from_HuTuKang/_Model\Model_for_GTMT/data_selected_1000_to_4000/Outer_Raceway_3000.mat'
dict_Outer_Raceway_4000 = u'F:/_project_bigdata_from_HuTuKang/_Model\Model_for_GTMT/data_selected_1000_to_4000/Outer_Raceway_4000.mat'
dict_Outer_Raceway_5000 = u'F:/_project_bigdata_from_HuTuKang/_Model\Model_for_GTMT_v2.0/data_selected_5000_to_8000/Outer_Raceway_5000.mat'
dict_Outer_Raceway_6000 = u'F:/_project_bigdata_from_HuTuKang/_Model\Model_for_GTMT_v2.0/data_selected_5000_to_8000/Outer_Raceway_6000.mat'
dict_Outer_Raceway_7000 = u'F:/_project_bigdata_from_HuTuKang/_Model\Model_for_GTMT_v2.0/data_selected_5000_to_8000/Outer_Raceway_7000.mat'
dict_Outer_Raceway_8000 = u'F:/_project_bigdata_from_HuTuKang/_Model\Model_for_GTMT_v2.0/data_selected_5000_to_8000/Outer_Raceway_8000.mat'
Outer_Raceway_1000 = io.loadmat(dict_Outer_Raceway_1000)['signal_save'].T
Outer_Raceway_2000 = io.loadmat(dict_Outer_Raceway_2000)['signal_save'].T
Outer_Raceway_3000 = io.loadmat(dict_Outer_Raceway_3000)['signal_save'].T
Outer_Raceway_4000 = io.loadmat(dict_Outer_Raceway_4000)['signal_save'].T
Outer_Raceway_5000 = io.loadmat(dict_Outer_Raceway_5000)['signal_save'].T
Outer_Raceway_6000 = io.loadmat(dict_Outer_Raceway_6000)['signal_save'].T
Outer_Raceway_7000 = io.loadmat(dict_Outer_Raceway_7000)['signal_save'].T
Outer_Raceway_8000 = io.loadmat(dict_Outer_Raceway_8000)['signal_save'].T
Outer_Raceway_data_list = [Outer_Raceway_1000, Outer_Raceway_2000, Outer_Raceway_3000, Outer_Raceway_4000, Outer_Raceway_5000, Outer_Raceway_6000, Outer_Raceway_7000, Outer_Raceway_8000]

# 信号截取段
signal_split = {
    # normal
    'Normal':
    [
        # normal_1000  3个来回  每段250000
        [[200000, 450000], [530000, 780000], [820000, 1070000], [1150000, 1400000], [1450000, 1700000], [1750000, 2000000]],
        # normal_2000  3个来回  每段100000
        [[230000, 330000], [380000, 480000], [530000, 630000], [700000, 800000], [850000, 950000], [1000000, 1100000]],
        # normal_3000  3个来回  每段90000
        [[190000, 280000], [280000, 370000], [390000, 480000], [480000, 570000], [600000, 690000], [690000, 780000]],
        # normal_4000  3个来回  每段50000
        [[220000, 270000], [300000, 350000], [380000, 430000], [450000, 500000], [530000, 580000], [610000, 660000]],

        # normal_5000  3个来回  每段50000
        [[637000, 687000], [700000, 750000], [765000, 815000], [830000, 880000], [890000, 940000], [960000, 1010000]],
        # normal_6000  3个来回  每段40000
        [[230000, 270000], [283000, 323000], [340000, 380000], [390000, 430000], [450000, 490000], [500000, 540000]],
        # normal_7000  3个来回  每段30000(只有5段,最后一段一分为二)
        [[220000, 250000], [270000, 300000], [315000, 345000], [360000, 390000], [410000, 425000], [425000, 440000]],
        # normal_8000  3个来回  每段30000
        [[215000, 245000], [260000, 290000], [300000, 330000], [345000, 375000], [385000, 415000], [425000, 455000]],
    ],
    # Inner Raceway
    'Inner_Raceway':
    [
        # normal_1000  3个来回  每段250000
        [[180000, 430000], [490000, 740000], [800000, 1050000], [1100000, 1350000], [1350000, 1600000], [1730000, 1980000]],
        # normal_2000  3个来回  每段100000
        [[150000, 250000], [300000, 400000], [460000, 560000], [620000, 720000], [760000, 860000], [930000, 1030000]],
        # normal_3000  3个来回  每段100000
        [[160000, 260000], [260000, 360000], [360000, 460000], [460000, 560000], [560000, 660000], [660000, 760000]],
        # normal_4000  3个来回  每段50000
        [[180000, 230000], [260000, 310000], [340000, 390000], [410000, 460000], [500000, 550000], [570000, 620000]],

        # Inner_Raceway_5000  3个来回  每段50000
        [[100000, 150000], [160000, 210000], [230000, 280000], [290000, 340000], [350000, 400000], [415000, 465000]],
        # Inner_Raceway_6000  3个来回  每段40000
        [[180000, 220000], [235000, 275000], [290000, 330000], [340000, 380000], [400000, 440000], [450000, 490000]],
        # Inner_Raceway_7000  3个来回  每段30000(只有5段,最后一段一分为二)
        [[110000, 140000], [150000, 180000], [200000, 230000], [250000, 280000], [295000, 310000], [310000, 325000]],
        # Inner_Raceway_8000  3个来回  每段30000
        [[86000, 116000], [130000, 160000], [170000, 200000], [215000, 245000], [255000, 285000], [295000, 325000]],
    ],
    # Outer Raceway
    "Outer_Raceway":
[
        # normal_1000  3个来回  每段250000
        [[220000, 470000], [520000, 770000], [830000, 1080000], [1150000, 1400000], [1460000, 1710000], [1770000, 2020000]],
        # normal_2000  3个来回  每段10000
        [[180000, 280000], [350000, 450000], [500000, 600000], [660000, 760000], [810000, 910000], [970000, 1070000]],
        # normal_3000  3个来回  每段100000
        [[220000, 320000], [320000, 420000], [420000, 520000], [520000, 620000], [620000, 720000], [720000, 820000]],
        # normal_4000  3个来回  每段50000
        [[150000, 200000], [230000, 280000], [310000, 360000], [390000, 440000], [470000, 520000], [550000, 600000]],

        # Outer_Raceway_5000  3个来回  每段50000
        [[200000, 250000], [265000, 315000], [330000, 380000], [395000, 445000], [450000, 500000], [520000, 570000]],
        # Outer_Raceway_6000  3个来回  每段40000
        [[250000, 290000], [305000, 345000], [355000, 395000], [410000, 450000], [470000, 510000], [520000, 560000]],
        # Outer_Raceway_7000  3个来回  每段30000(只有5段,最后一段一分为二)
        [[200000, 230000], [250000, 280000], [300000, 330000], [350000, 380000], [390000, 405000], [405000, 420000]],
        # Outer_Raceway_8000  3个来回  每段30000
        [[314000, 344000], [350000, 380000], [395000, 425000], [435000, 465000], [480000, 510000], [520000, 550000]],
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
    print(" array.shape",array.shape)  # (3810000, 1)
    array_enhance = np.empty([length , multip])  # (2048,43200)
    array_len = len(array) - length
    overlap = int(array_len/(multip - 1)) # 窗口滑移时的重叠长度
    print("overlap: ", overlap)
    for i in range(int(multip/2)):
        array_enhance[:, i] = array[(overlap * i) : (overlap * i + length),0]  # 从前往后写入数据
        #print("array[(overlap * i) : (overlap * i + length)].shape",array[(overlap * i) : (overlap * i + length)].shape)  # 2048
        #print("array_enhance[:, i].shape",array_enhance[:, i].shape)
        array_enhance[:, multip -i -1] = array[(array_len - overlap * i): (array_len - overlap * i + length),0]  # 从后往前写入数据
    if multip % 2 == 1:
        array_enhance[:, int(multip / 2)] = array[int(array_len / 2) : int(array_len / 2 + length),0]  # # 如果multip是奇数则中间再插补一个2048的数据
    print("array_enhance.T.shape",array_enhance.T.shape)
    return array_enhance.T


def main_signal_process(data_list,signal_split_key,one_hot_label ):
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
    for index, data in enumerate(data_list):               # 对normal中4中进给速度的数据循环
        print("index",index,'\n',"data.shape:",data.shape) # 没进行切分时的原始长度，(2334720, 1)
        per_feed_data = []                                 # 某种轴承故障状态下，每一种进给速度下的数据

        for item in signal_split_key[index]: # 当index == 0，对normal中第1种速度的6个片段循环，item返回6个区间
            print("index",index)
            print("item:",item)                                           # 如 [520000, 560000]
            vibration = data[item[0] : item[1]]                           # 数据截取，如 data(520000 : 560000)
            print("vibration.shape",vibration.shape)                      # 每一小段数据的长度，(30000, 1)
            enhanced_vibration = data_enhance(vibration, split_length, enhance_num)  # 数据增强
            print("enhanced vibration data.shape", enhanced_vibration.shape)   # (300, 2048)

            per_feed_data.append(enhanced_vibration)                          #

        print("per_feed_data.shape",np.array(per_feed_data).shape)         # 纵向堆叠，每种转速下的6段数据(6, 300, 2048)
        per_feed_data_array = np.concatenate(per_feed_data,axis=0)
        print("per_feed_data_array.shape",np.array(per_feed_data_array).shape)        # (1800, 2048)
        per_fault_data.append(per_feed_data_array)

    print("per_fault_data.shape",np.array(per_fault_data).shape)         # (8, 1800, 2048) 所有转速种类堆叠，8种转速
    per_fault_data_array = np.concatenate(per_fault_data, axis=0)
    print("per_fault_data_array.shape",per_fault_data_array.shape)       # (14400, 2048)

    label         = np.array(one_hot_label * len(per_fault_data_array))
    label_one_hot = label.reshape(len(per_fault_data_array), -1)
    print("label_one_hot.shape",label_one_hot.shape)                     # (14400, 3)

    num_sample_per_fault = len(per_fault_data_array)
    index_permutation = np.arange(num_sample_per_fault)
    np.random.shuffle(index_permutation)
    shuffled_data = per_fault_data_array[index_permutation][          : num_sample_per_fault]
    print("shuffled_data.shape",shuffled_data.shape)                     # (14400, 2048)

    return shuffled_data, label_one_hot



'''================================================== 主函数调用 ============================================'''
#  数据预处理
print('='*100)
# Normal
data_normal,        label_normal        = main_signal_process(data_list=normal_data_list,                 # 4个文件的数据列表
                                                signal_split_key=signal_split['Normal'],    # 4个 normal轴承的6个片段（4*6*2）
                                                one_hot_label=normal_label )                # norm_label = [1, 0, 0]

# Inner_Raceway
data_Inner_Raceway, label_Inner_Raceway = main_signal_process(data_list=Inner_Raceway_data_list,  # 4个文件的数据列表
                                             signal_split_key=signal_split['Inner_Raceway'],      # 4个 normal轴承的6个片段（4*6*2）
                                             one_hot_label=Inner_Raceway_label )                  # Inner_Raceway_label = [0, 1, 0]

# Outer_Raceway
data_Outer_Raceway, label_Outer_Raceway = main_signal_process(data_list=Outer_Raceway_data_list,  # 4个文件的数据列表
                                             signal_split_key=signal_split['Outer_Raceway'],      # 4个 normal轴承的6个片段（4*6*2）
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












