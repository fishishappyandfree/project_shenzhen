# coding=utf-8
# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

import os
from scipy import io
import time
import numpy
from matplotlib import pyplot as plt
from scipy import signal
from nptdms import TdmsFile


# 故障部位标签
norm_label = [0, 0, 1]
in_600_label = [0, 1, 0]
out_600_label = [1, 0, 0]

# TDMS文件路径
# Norm data 文件路径
# tdms_path_norm_10000_0 = r"D:\_project_bigdata_from_HuTuKang\_experiment_data_analysis\data_from_machine_tool_3\bearing_x_axis_normal\normal_bearing_num_3\new_all_feed_x_10000-9-29-06.tdms"
# tdms_path_norm_10000_1 = r"D:\_project_bigdata_from_HuTuKang\_experiment_data_analysis\data_from_machine_tool_3\bearing_x_axis_normal\normal_bearing_num_4\x_new_num3_feed10000-14-55-30.tdms"
# tdms_path_norm_20000_0 = r"D:\_project_bigdata_from_HuTuKang\_experiment_data_analysis\data_from_machine_tool_3\bearing_x_axis_normal\normal_bearing_num_3\new_all_feed_x_20000-9-31-21.tdms"
# tdms_path_norm_20000_1 = r"D:\_project_bigdata_from_HuTuKang\_experiment_data_analysis\data_from_machine_tool_3\bearing_x_axis_normal\normal_bearing_num_4\x_new_num3_feed20000-14-56-33.tdms"
# tdms_path_norm_list = [tdms_path_norm_10000_0, tdms_path_norm_10000_1,
                       tdms_path_norm_20000_0, tdms_path_norm_20000_1]
# tdms_path_norm_list_name = ['tdms_path_norm_10000_0', 'tdms_path_norm_10000_1',
                            'tdms_path_norm_20000_0', 'tdms_path_norm_20000_1']
# norm_label = [0, 0, 1]

# inner 0.6-0.02
tdms_path_in_600_10000_0 = r"D:\_project_bigdata_from_HuTuKang\_experiment_data_analysis\data_from_machine_tool_3\bearing_x_axis_width_0.6mm_depth_0.02mm_num_1\data_not_cutting\degree_0\x_0.6_0.02_num4_feed10000-19-03-27.tdms"
tdms_path_in_600_10000_1 = r"D:\_project_bigdata_from_HuTuKang\_experiment_data_analysis\data_from_machine_tool_3\bearing_x_axis_width_0.6mm_depth_0.02mm_num_1\data_not_cutting\degree_180\x_0.6_0.02_num5_feed10000-15-08-05.tdms"
tdms_path_in_600_20000_0 = r"D:\_project_bigdata_from_HuTuKang\_experiment_data_analysis\data_from_machine_tool_3\bearing_x_axis_width_0.6mm_depth_0.02mm_num_1\data_not_cutting\degree_0\x_0.6_0.02_num4_feed20000-19-04-34.tdms"
tdms_path_in_600_20000_1 = r"D:\_project_bigdata_from_HuTuKang\_experiment_data_analysis\data_from_machine_tool_3\bearing_x_axis_width_0.6mm_depth_0.02mm_num_1\data_not_cutting\degree_180\x_0.6_0.02_num5_feed20000-15-05-30.tdms"
tdms_path_in_600_list = [tdms_path_in_600_10000_1, tdms_path_in_600_10000_1,
                         tdms_path_in_600_20000_0, tdms_path_in_600_20000_1]
tdms_path_in_600_list_name = ['tdms_path_in_600_10000_0', 'tdms_path_in_600_10000_1',
                              'tdms_path_in_600_20000_0', 'tdms_path_in_600_20000_1']
# in_600_label = [0, 1, 0]

# outer 0.6-0.02
tdms_path_out_600_10000_0 = r"D:\_project_bigdata_from_HuTuKang\_experiment_data_analysis\data_from_machine_tool_3\beraing_x_axis_width_0.6mm_depth_0.02mm_num_2\data_not_cutting\degree_0\x_out_0.6_0.02_num6_feed10000-15-07-23.tdms"
tdms_path_out_600_10000_1 = r"D:\_project_bigdata_from_HuTuKang\_experiment_data_analysis\data_from_machine_tool_3\beraing_x_axis_width_0.6mm_depth_0.02mm_num_2\data_not_cutting\degree_180\x_out_deg180_0.6_0.02_num6_feed10000-19-53-33.tdms"
tdms_path_out_600_20000_0 = r"D:\_project_bigdata_from_HuTuKang\_experiment_data_analysis\data_from_machine_tool_3\beraing_x_axis_width_0.6mm_depth_0.02mm_num_2\data_not_cutting\degree_0\x_out_0.6_0.02_num6_feed20000-15-06-31.tdms"
tdms_path_out_600_20000_1 = r"D:\_project_bigdata_from_HuTuKang\_experiment_data_analysis\data_from_machine_tool_3\beraing_x_axis_width_0.6mm_depth_0.02mm_num_2\data_not_cutting\degree_180\x_out_deg180_0.6_0.02_num6_feed20000-19-52-59.tdms"
tdms_path_out_600_list = [tdms_path_out_600_10000_0, tdms_path_out_600_10000_1,
                          tdms_path_out_600_20000_0, tdms_path_out_600_20000_1]
tdms_path_out_600_list_name = ['tdms_path_out_600_10000_0', 'tdms_path_out_600_10000_1',
                               'tdms_path_out_600_20000_0', 'tdms_path_out_600_20000_1']
# out_600_label = [1, 0, 0]

# 结果保存文件夹，保存文件格式为.mat文件，也可以保存为其他文件,只需要修改对应的保存函数即可
#mat_save_path = r"D:\_project_bigdata_from_HuTuKang\_experiment_data_analysis\data_from_machine_tool_3\mat"

mat_save_path = r"D:\_project_bigdata_from_HuTuKang\_experiment_data_analysis\data_from_machine_tool_3\mat"



# 传感器通道名，需要修改
key = ["/'未命名'/'cDAQ9189-1D71297Mod6/ai2'",   # feed_x_axis channel number
       "/'未命名'/'cDAQ9189-1D71297Mod4/ai1'"]   # feed_y_axis channel number


# 信号截取段
signal_split = {
    # normal
    'normal':
    [
        # new_all_feed_x_10000-9-29-06.tdms
        [[235000, 295000], [305000, 365000], [385000, 445000], [
            455000, 515000], [535000, 595000], [605000, 665000]],
        # x_new_num3_feed10000-14-55-30.tdms
        [[80000, 130000], [135000, 185000], [200000, 250000], [
            260000, 310000], [320000, 370000], [380000, 430000]],
        # new_all_feed_x_20000-9-31-21.tdms
        [[60000, 90000], [95000, 125000], [145000, 165000], [
            173000, 203000], [210000, 240000], [250000, 280000]],
        # x_new_num3_feed20000-14-56-33.tdms
        [[60000, 85000], [90000, 115000], [123000, 148000], [
            153000, 178000], [183000, 208000], [213000, 238000]]
    ],

    # inner
    'inner_600':
    [
        # x_0.6_0.02_num4_feed10000-19-03-27.tdms
        [[30000, 80000], [90000, 140000], [150000, 200000], [
            215000, 265000], [270000, 320000], [340000, 390000]],
        # x_0.6_0.02_num5_feed10000-15-08-05.tdms
        [[150000, 200000], [210000, 260000], [270000, 320000], [
            335000, 385000], [395000, 445000], [455000, 505000]],
        # x_0.6_0.02_num4_feed20000-19-04-34.tdms
        [[38000, 63000], [70000, 95000], [100000, 125000], [
            130000, 155000], [160000, 185000], [192000, 217000]],
        # x_0.6_0.02_num5_feed20000-15-05-30.tdms
        [[75000, 100000], [105000, 130000], [135000, 160000], [
            165000, 190000], [197000, 222000], [228000, 253000]]
    ],
    # outer
    "outer_600":
    # x_out_0.6_0.02_num6_feed10000-15-07-23.tdms
    [
        [[70000, 120000], [130000, 180000], [190000, 240000], [
            250000, 300000], [315000, 365000], [380000, 430000]],
        # x_out_deg180_0.6_0.02_num6_feed10000-19-53-33.tdms
        [[40000, 90000], [105000, 155000], [165000, 215000], [
            225000, 275000], [290000, 340000], [350000, 400000]],
        # x_out_0.6_0.02_num6_feed20000-15-06-31.tdms
        [[60000, 85000], [90000, 115000], [123000, 148000], [
            152000, 177000], [182000, 207000], [215000, 240000]],
        # x_out_deg180_0.6_0.02_num6_feed20000-19-52-59.tdms
        [[32000, 57000], [63000, 88000], [95000, 120000], [
            125000, 150000], [155000, 180000], [187000, 204799]]
    ]
}

# 采样频率
Fs = 25600
# 截取段长度
N = 2048
# 截取倍数
T = 300

# 获取信号数据
def get_signal(tdms_path, channel_num, lower_num, upper_num):    # 截取信号开头与结尾
     #       （路径， key[0]="/'未命名'/'cDAQ9189-1D71297Mod5/ai3'",item[0]=235000, item[1]=295000])
    if type(channel_num) != 'str':
        channel_num = str(channel_num)    #强制类型转换
    print(tdms_path)
    file = TdmsFile(tdms_path)
    #print(type(file))
   # print(channel_num.encode(encoding='utf-8'))
    #print(file.objects)
    # keyList=[]
    # for i in file.objects.keys():
    #     #print(type(i.encode('utf-8')))
    #     keyList.append(i.encode('utf-8'))
    # print(len(keyList))
    # for j in range(len(keyList)):
    #     #print(keyList)
    #     file.objects[keyList[j]]= file.objects.pop(file.objects[file.objects.keys()[j]])
    # for i in file.objects.keys():
    #     print(i)
    # tdms_data = file.objects[channel_num].data  #取出数据
    #  print(file.object(key[0]))
    tdms_data = file.objects[channel_num.decode(encoding='utf-8')].data  #取出数据
    if lower_num * upper_num == 0:            #如果 有一个数为0，则截取整段信号
        lower_num = 0
        upper_num = len(tdms_data)
    else:
        if lower_num > upper_num:             # 调整大小顺序
            lower_num, upper_num = upper_num, lower_num
    return tdms_data[lower_num: upper_num]
# 绘制时域图
def plot_waveform_func(array, para_list=['waveform', 'time', 'frequency'], path_2_save=None, lower_num=None, upper_num=None, other_smg=None):
    assert len(para_list) == 3

    fig, ax = plt.subplots()
    ax.plot(array)
    ax.set_title(para_list[0])     # 标题 'waveform'
    ax.set_xlabel(para_list[1])    # X轴'time'
    ax.set_ylabel(para_list[2])
    plt.tight_layout()             # tight_layout会自动调整子图参数，使之填充整个图像区域。
    if lower_num > upper_num:
        raise ImportError('Error lower_num and upper_num! ')

    file_name = '_' + other_smg + \
        para_list[0] + '_' + str(lower_num) + '_' + \
        str(lower_num) + '_' + str(time.time())
    plt.savefig(os.path.join(path_2_save, file_name + '.png'), dpi=600)
    plt.close()   # Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed
    # plt.savefig(os.path.join(path_2_save, file_name + '.svg'))
# 数据增强
def enhance_ave(array, N, T):
    '''
    平均截取
    # 截取段长度
    N = 2048
    # 截取倍数
    T = 300

    Output:
    返回扩充后的N*T数组
    '''
    # print(type(array))
    # print(array.shape)
    # print('array[: 10] is :')
    # print(array[:10])
    array_empty = numpy.empty([N, T])
    arr_len = len(array)   # arr_len =60000 每个数据小片段长度
    length = arr_len - N   # 60000-2048=57952
    interval = int(length / (T - 1))  # 间隔interval =57952/299=193.81

    for i in range(int(T / 2)): # i from 0~149
        array_empty[:, i] = array[(interval * i): (interval * i + N)]
    #   array_empty[:, 1] = array[(193.81 * 1): (193.81 * 1 + 300)]=array[193:493]
        array_empty[:, T - i -1] = array[(arr_len - interval * i - N): (arr_len - interval * i)]
    if T % 2 == 1:
        array_empty[:, int(T / 2)] = array[int(length / 2) : int(length / 2 + N)]
    print("array_empty.T.shape\n",array_empty.T.shape)   #(300*2048)
    print("arr_len",arr_len)
    return array_empty.T

# 生成对应的标签数据
def gen_label(array, label):
    '''
    # 根据生成的Data，生成对应的Label数据, 生成的Label的深度为Enlarge.T

    Input:
    label: list, Label的生成模板

    Output:
    生成对应的多维Label数组
    '''
    label = label.tolist() if type(label) == numpy.ndarray else label   # 类型转换
    # depth个label堆叠
    label_long = numpy.array(label * len(array))
    return label_long.reshape([len(array), -1])

'''
def plot_stft_func(array, para_list=['STFT of vib', 'time', 'frequency'], path_2_save=None):
    # 强制类型转变

    print(array.shape)
    assert len(para_list) == 3
    fig, ax = plt.subplots()
    ax.plot(array)
    # plt.show()

    f, t, Zxx = signal.stft(x_axis_vib, Fs, 'hann', nperseg)
    fig, ax = plt.subplots()
    im = ax.pcolormesh(t, f, abs(Zxx), edgecolors='face', shading='flat')
    ax.set_title(para_list[0])
    ax.set_xlabel(para_list[1])
    ax.set_ylabel(para_list[2])
    c_bar = fig.colorbar(im, ax=ax)
    plt.tight_layout()

    file_name = para_list[0] if lower_num * upper_num == 0 else para_list[0] + '_' + str(lower_num) + '_' + str(lower_num)
    plt.savefig(os.path.join(path_2_save, file_name + '.png'), dpi=600)
    # plt.savefig(os.path.join(path_2_save, file_name + '.svg'))
    # plt.show()
'''

def signal_process_meta(array, N, T, label):
  #       (X_feed数据,截取段长度N = 2048,截取倍数T = 300,in_600_label = [0, 1, 0])

    # print('array type is:', type(array))
    # print('item type is:', type(array[0]))
    # print(type(array) != 'list')
    if type(array) != list:
        array = [array]        # list 转化为 array
    # print('item type is:', type(array[0]))
    # print(label)
    array_enhance = [enhance_ave(item, N, T) for item in array]
    label_enhance = gen_label(array_enhance[0], label)

    print("array_enhance.shape\n",numpy.array(array_enhance).shape) # (1, 300, 2048)
    print("label_enhance.shape\n",numpy.array(label_enhance).shape)  # (300, 3)  [0 0 1].........

    return array_enhance, label_enhance

def rearrange_save2mat(array, save2path=None):    # 重排
    array_rearrange_0 = numpy.concatenate([array[index][0] for index in range(len(array))], axis=0)   # 数组拼接，安装列堆叠
    array_rearrange_1 = numpy.concatenate([array[index][1] for index in range(len(array))], axis=0)
    array_rearrange_2 = numpy.concatenate([array[index][2] for index in range(len(array))], axis=0)
    array_rearrange_3 = numpy.concatenate([array[index][3] for index in range(len(array))], axis=0)
    array_rearrange_4 = numpy.concatenate([array[index][4] for index in range(len(array))], axis=0)
    array_rearrange_5 = numpy.concatenate([array[index][5] for index in range(len(array))], axis=0)
    # print('array_rearrange_0 shape: ,', )
    # return [array_rearrange_0, array_rearrange_1, array_rearrange_2, array_rearrange_3, array_rearrange_4, array_rearrange_5]
    mat_file = {
        'data0' :array_rearrange_0,    # 1200*2048
        'data1' :array_rearrange_1,    # 1200*2048
        'data2' :array_rearrange_2,    # 1200*2048
        'data3' :array_rearrange_3,    # 1200*2048
        'data4' :array_rearrange_4,    # 1200*2048
        'data5' :array_rearrange_5     # 1200*2048
    }
    io.savemat(save2path, mat_file)
    return [array_rearrange_0, array_rearrange_1, array_rearrange_2, array_rearrange_3, array_rearrange_4, array_rearrange_5]

def signal_process(tdms_path_list, signal_key, other_smg, N, T, label_list, mat_save_path, mat_name_list):
    # print(label_list)
    #assert 确保程序没错，若出错，会输出"Error Input of tdms_path_list or signal_key!"
    #assert len(tdms_path_list) == len(signal_key), print("Error Input of tdms_path_list or signal_key! ")
    assert len(mat_name_list) == 2

    # vib_x_train_set_list, vib_x_test_set_list, vib_x_train_label_list, vib_x_test_label_list = [], [], [], []
    vib_x_set_list, label_set_list = [], []
    for index, path in enumerate(tdms_path_list):   #enumerate会将该数据对象组合为一个索引序列，同时列出数据和数据下标
        # vib_x_train_set, vib_x_test_set, vib_x_train_label, vib_x_test_label = [], [], [], []
        vib_x_set, label_set = [], []
        # train_set, test_set, train_label, test_label = [], [], [], []
        print(path)
        for item in signal_key[index]:  # index 表示路径号  第0个路径-第3个路径
            # 当 index==0, item=如下
            # [235000, 295000]
            # [305000, 365000]
            # [385000, 445000]
            # [455000, 515000]
            # [535000, 595000]
            # [605000, 665000]
            print(key[0])
            #获取对应路径，对应通道的，对应起始、结束区间的数据
            vib_x = get_signal(path, key[0], item[0], item[1])  # X相振动信号
            #                （路径， key[0]="/'未命名'/'cDAQ9189-1D71297Mod5/ai3'",item[0]=235000, item[1]=295000])
            vib_y = get_signal(path, key[1], item[0], item[1])   # Y相振动信号

            array, label = signal_process_meta([vib_x], N, T, label_list)
            #                                 (X_feed数据,截取段长度N = 2048,截取倍数T = 300,in_600_label = [0, 1, 0])
            vib_x_set.append(array[0])   # 数组添加
            label_set.append(label)
        #  4个 300*2048 连接起来  形成 1200*2048的数组

        vib_x_set_list.append(vib_x_set)
        label_set_list.append(label_set)
    
    vib_mat = rearrange_save2mat(vib_x_set_list, save2path=os.path.join(mat_save_path, mat_name_list[0]))
    label_mat = rearrange_save2mat(label_set_list, save2path=os.path.join(mat_save_path, mat_name_list[1]))
    # array_list_0 = [vib_x_set_list[index][0] for index in range(len(vib_x_set_list))]
    # array_list_1 = [vib_x_set_list[index][1] for index in range(len(vib_x_set_list))]
    # array_list_2 = [vib_x_set_list[index][2] for index in range(len(vib_x_set_list))]
    # array_list_3 = [vib_x_set_list[index][3] for index in range(len(vib_x_set_list))]
    # array_list_4 = [vib_x_set_list[index][4] for index in range(len(vib_x_set_list))]
    # array_list_5 = [vib_x_set_list[index][5] for index in range(len(vib_x_set_list))]

    # label_list_0 = [label_set_list[index][0] for index in range(len(label_set_list))]
    # label_list_1 = [label_set_list[index][1] for index in range(len(label_set_list))]
    # label_list_2 = [label_set_list[index][2] for index in range(len(label_set_list))]
    # label_list_3 = [label_set_list[index][3] for index in range(len(label_set_list))]
    # label_list_4 = [label_set_list[index][4] for index in range(len(label_set_list))]
    # label_list_5 = [label_set_list[index][5] for index in range(len(label_set_list))]

            # microphone = get_signal(path, key[2], item[0],item[1])
            # plot_waveform_func(vib_x, ['vib_x waveform', 'time', 'amp'],
            #                    h5_save_path, item[0], item[1], other_smg[index])
            # plot_waveform_func(vib_y, ['vib_y waveform', 'time', 'amp'],
            #                    h5_save_path, item[0], item[1], other_smg[index])
            # plot_waveform_func(microphone, ['microphone waveform', 'time', 'amp'], h5_save_path, item[0],item[1])

        # plot_waveform_func(vib_x, para_list=['vib_x waveform', 'time', 'amp'], signal_key[index][0],signal_key[index][1])

    # return vib_x_set_list, label_set_list
    return vib_mat, label_mat
    # return [array_list_0, array_list_1, array_list_2, array_list_3, array_list_4, array_list_5], [label_list_0, label_list_1, label_list_2, label_list_3, label_list_4, label_list_5]
    # return rearrange_save2h5(vib_x_set_list, save2path=None, filename=None), rearrange_save2h5(label_set_list, save2path=None, filename=None)


'''==================================================函数调用============================================'''
#  数据预处理
vib_x_norm_set_list, label_norm_set_list = signal_process(tdms_path_list=tdms_path_norm_list,  # 4个文件的路径列表
                                           signal_key=signal_split['normal'],      # 4个 normal轴承的6个片段（4*6*2）
                                           other_smg=tdms_path_norm_list_name,    # 4个文件的名称
                                           N=N,                                 # 截取段长度 N=2048
                                           T=T,                                 # 截取倍数 T=300
                                           label_list=norm_label,               # norm_label = [0, 0, 1]
                                           mat_save_path=mat_save_path,         # mat文件的保存路径
                                           mat_name_list=['norm_data.mat', 'norm_label.mat'])   #保存的文件名

print("vib_x_norm_set_list:\n",vib_x_norm_set_list,'\n',"label_norm_set_list:\n", label_norm_set_list)

print(numpy.array(vib_x_norm_set_list).shape,'\n',numpy.array(label_norm_set_list).shape)

vib_x_in_600_set_list, label_in_600_set_list = signal_process(tdms_path_list=tdms_path_in_600_list,
                                            signal_key=signal_split['inner_600'], 
                                            other_smg=tdms_path_in_600_list_name,
                                            N=N,
                                            T=T,
                                            label_list=in_600_label,
                                            mat_save_path=mat_save_path, 
                                            mat_name_list=['inner_600_data.mat', 'inner_600_label.mat'])

vib_x_out_600_set_list, label_out_600_set_list = signal_process(tdms_path_list=tdms_path_out_600_list,
                                            signal_key=signal_split['outer_600'], 
                                            other_smg=tdms_path_out_600_list_name,
                                            N=N,
                                            T=T,
                                            label_list=out_600_label,
                                            mat_save_path=mat_save_path, 
                                            mat_name_list=['outer_600_data.mat', 'outer_600_label.mat'])

x_axis_vib = get_signal(tdms_path, key[0], lower_num, upper_num)
y_axis_vib = get_signal(tdms_path, key[1], lower_num, upper_num)
microphone = get_signal(tdms_path, key[2], lower_num, upper_num)


# 画图
plot_waveform_func(x_axis_vib, para_list=[
           'STFT of x-axis vibration', 'time', 'frequency'], path_2_save=path_2_save)
#plot_waveform_func(f_y, t_y, Zxx_y, para_list=['STFT of y-axis vibration', 'time', 'frequency'], path_2_save=path_2_save)
#plot_waveform_func(f_m, t_m, Zxx_m, para_list=['STFT of microphone', 'time', 'frequency'], path_2_save=path_2_save)
