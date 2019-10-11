# coding=utf-8
import os
from scipy import io
import numpy
from nptdms import TdmsFile


# 故障部位标签
norm_label = [0, 0, 1]
in_600_label = [0, 1, 0]
out_600_label = [1, 0, 0]

# TDMS文件路径
# Norm data 文件路径
tdms_path_norm_10000_0 = r"G:\js\results\new_all_feed_x_10000-9-29-06.tdms"
tdms_path_norm_10000_1 = r"G:\js\results\x_new_num3_feed10000-14-55-30.tdms"
tdms_path_norm_20000_0 = r"G:\js\results\new_all_feed_x_20000-9-31-21.tdms"
tdms_path_norm_20000_1 = r"G:\js\results\x_new_num3_feed20000-14-56-33.tdms"
tdms_path_norm_list = [tdms_path_norm_10000_0, tdms_path_norm_10000_1,
                       tdms_path_norm_20000_0, tdms_path_norm_20000_1]
tdms_path_norm_list_name = ['tdms_path_norm_10000_0', 'tdms_path_norm_10000_1',
                            'tdms_path_norm_20000_0', 'tdms_path_norm_20000_1']
# norm_label = [0, 0, 1]

# inner 0.6-0.02
tdms_path_in_600_10000_0 = r"G:\js\results\x_0.6_0.02_num4_feed10000-19-03-27.tdms"
tdms_path_in_600_10000_1 = r"G:\js\results\x_0.6_0.02_num5_feed10000-15-08-05.tdms"
tdms_path_in_600_20000_0 = r"G:\js\results\x_0.6_0.02_num4_feed20000-19-04-34.tdms"
tdms_path_in_600_20000_1 = r"G:\js\results\x_0.6_0.02_num5_feed20000-15-05-30.tdms"
tdms_path_in_600_list = [tdms_path_in_600_10000_1, tdms_path_in_600_10000_1,
                         tdms_path_in_600_20000_0, tdms_path_in_600_20000_1]
tdms_path_in_600_list_name = ['tdms_path_in_600_10000_0', 'tdms_path_in_600_10000_1',
                              'tdms_path_in_600_20000_0', 'tdms_path_in_600_20000_1']
# in_600_label = [0, 1, 0]

# outer 0.6-0.02
tdms_path_out_600_10000_0 = r"G:\js\results\x_out_0.6_0.02_num6_feed10000-15-07-23.tdms"
tdms_path_out_600_10000_1 = r"G:\js\results\x_out_deg180_0.6_0.02_num6_feed10000-19-53-33.tdms"
tdms_path_out_600_20000_0 = r"G:\js\results\x_out_0.6_0.02_num6_feed20000-15-06-31.tdms"
tdms_path_out_600_20000_1 = r"G:\js\results\x_out_deg180_0.6_0.02_num6_feed20000-19-52-59.tdms"
tdms_path_out_600_list = [tdms_path_out_600_10000_0, tdms_path_out_600_10000_1,
                          tdms_path_out_600_20000_0, tdms_path_out_600_20000_1]
tdms_path_out_600_list_name = ['tdms_path_out_600_10000_0', 'tdms_path_out_600_10000_1',
                               'tdms_path_out_600_20000_0', 'tdms_path_out_600_20000_1']
# out_600_label = [1, 0, 0]

# 结果保存文件夹，保存文件格式为.mat文件，也可以保存为其他文件,只需要修改对应的保存函数即可
#mat_save_path = r"D:\_project_bigdata_from_HuTuKang\_experiment_data_analysis\data_from_machine_tool_3\mat"

mat_save_path = r"G:\js\code\model_new\mat"



# 传感器通道名，需要修改
key = ["/'未命名'/'cDAQ9189-1D71297Mod5/ai3'",   # feed_x_axis channel number
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


# 读取指定工况信号区间的数据
def get_signal(tdms_path, channel_num, lower_num, upper_num):
    '''
    读取指定工况信号区间的数据
    :param tdms_path: 文件路径
    :param channel_num: 通道名
    :param lower_num: 截取区间左
    :param upper_num: 截取区间右
    :return: 指定区间数据
    '''
     #       （路径， key[0]="/'未命名'/'cDAQ9189-1D71297Mod5/ai3'",item[0]=235000, item[1]=295000])
    if type(channel_num) != 'str':
        channel_num = str(channel_num)    #强制类型转换
    file = TdmsFile(tdms_path)
    tdms_data = file.objects[channel_num].data  #取出数据
    if lower_num * upper_num == 0:            #如果 有一个数为0，则截取整段信号
        lower_num = 0
        upper_num = len(tdms_data)
    else:
        if lower_num > upper_num:             # 调整大小顺序
            lower_num, upper_num = upper_num, lower_num
    return tdms_data[lower_num: upper_num]


# 数据增强  (T,N)矩阵
def enhance_ave(array, N, T):
    '''
    将每一段加工信号数据（len(array),）扩充为T*N
    :param array: 截取获得的一维数据
    :param N: 截取段长度 N=2048
    :param T: 截取倍数 T=300
    :return: T*N矩阵
    '''
    array_empty = numpy.empty([N, T]) # 新建 N*T 的0矩阵
    arr_len = len(array)   # arr_len =60000 每个数据小片段长度,看signal_split
    length = arr_len - N   # 60000-2048=57952
    interval = int(length / (T - 1))  # 间隔interval =int(57952/299)=int(193.81)=193

    # 此处获取的2048个点，是每次向右走193个点取得的，每相邻的两个片段只有193个点不一样
    for i in range(int(T / 2)): # i from 0~149
        array_empty[:, i] = array[(interval * i): (interval * i + N)]
    #   array_empty[:, 1] = array[(193 * 1): (193 * 1 + 2048)]=array[193:2048+193]
        array_empty[:, T - i -1] = array[(arr_len - interval * i - N): (arr_len - interval * i)]
    if T % 2 == 1:
        array_empty[:, int(T / 2)] = array[int(length / 2) : int(length / 2 + N)]
    print("array_empty.T.shape\n",array_empty.T.shape)   #(300*2048)
    print("arr_len",arr_len)
    return array_empty.T


# 生成对应维度的标签数据  (T,len(label))
def gen_label(array, label):
    '''
    获得和样本特征对应维度的标签矩阵
    :param array: 匹配的样本特征 T*N
    :param label: 每个样本对应的标签
    :return: T,len(label) 标签矩阵
    '''
    label = label.tolist() if type(label) == numpy.ndarray else label   # 类型转换
    # label堆叠
    label_long = numpy.array(label * len(array)) # len(array)==T
    return label_long.reshape([len(array), -1])


# 扩充后对应维度的加工信号数据(T,N)矩阵，标签数据(T,len(label))
def signal_process_meta(array, N, T, label):
    '''
    获取扩充后对应维度的加工信号数据和标签数据
    :param array:
    :param N: 截取段长度 N=2048
    :param T: 截取倍数 T=300
    :param label: 标签值，nomal,in,out：one hot编码
    :return: 扩充后对应维度的加工信号数据，标签数据
    '''
    # (X_feed数据,截取段长度N = 2048,截取倍数T = 300,in_600_label = [0, 1, 0])

    array_enhance = enhance_ave(array, N, T)
    label_enhance = gen_label(array_enhance, label)

    print("array_enhance.shape\n",numpy.array(array_enhance).shape) # (300, 2048)
    print("label_enhance.shape\n",numpy.array(label_enhance).shape)  # (300, 3)  [[0 0 1],.........]

    return array_enhance, label_enhance


def rearrange_save2mat(array, N, save2path=None):
    '''
    改变数据维度，并按照，有加工时段取6段数据，每段数据为某种故障类型4种工况，保存
    :param array: 四维数据 样本：(4,6,300,2048)  标签：(4,6,300,3)
    :param N: 最低维度数，可理解成二维里的列数
    :param save2path: mat保存地址
    :return: 6个样本集（1200,2048），6个标签集（1200,3）
    '''
    list_data=['data0', 'data1', 'data2', 'data3', 'data4', 'data5']
    mat_file={}
    print(array.shape[1])
    for i in range(array.shape[1]):
        mat_file[list_data[i]] = array[:,i,:,:].reshape(-1, N)

    io.savemat(save2path, mat_file)
    return list(mat_file.values())


def signal_process(tdms_path_list, signal_key, other_smg, N, T, label_list, mat_save_path, mat_name_list):
    '''
    读取指定TDMS文件，将信号数据扩充为(T,N),标签数据为(T,len(label))，按照指定格式保存数据
    :param tdms_path_list: 某种故障类型中TDMS数据文件路径
    :param signal_key: 某种故障类型中截取有效数据段
    :param other_smg: 文某种故障类型中件路径变量的名称
    :param N: 截取段长度 N=2048
    :param T: 截取倍数 T=300
    :param label_list: 标签值，nomal,in,out：one hot编码
    :param mat_save_path: mat文件保存路径
    :param mat_name_list: mat文件名命名方式
    :return: 6个样本集（1200,2048），6个标签集（1200,3）
    '''
    vib_x_set_list, label_set_list = [], []
    for index, path in enumerate(tdms_path_list):   #enumerate会将该数据对象组合为一个索引序列，同时列出数据和数据下标，index:0,1,2,3
        vib_x_set, label_set = [], []
        for item in signal_key[index]: # signal_key[index]遍历某种故障类型下每种工况下的6段数据
            #获取对应路径，对应通道的，对应起始、结束区间的数据
            vib_x = get_signal(path, key[0], item[0], item[1])  # X相振动信号
            #                （路径， key[0]="/'未命名'/'cDAQ9189-1D71297Mod5/ai3'",item[0]=235000, item[1]=295000])

            #由原加工信号获取扩充后对应维度的加工信号数据(T,N)矩阵，标签数据(T,len(label))
            array, label = signal_process_meta(vib_x, N, T, label_list)
            #                                 (X_feed数据,截取段长度N = 2048,截取倍数T = 300,in_600_label = [0, 1, 0])

            #np.array(vib_x_set).shape == (6,300,2048)
            vib_x_set.append(array)   # vib_x_set每种故障类型每个加工工况的6组训练数据
            # np.array(label_set).shape == (6, 300, 3)
            label_set.append(label)

        # np.array(vib_x_set_list).shape == (4,6,300,2048)
        vib_x_set_list.append(vib_x_set) # vib_x_set_list每种故障类型4个加工工况的训练数据
        # np.array(label_set_list).shape == (4,6,300,3)
        label_set_list.append(label_set)
    
    vib_mat = rearrange_save2mat(numpy.array(vib_x_set_list), N, save2path=os.path.join(mat_save_path, mat_name_list[0]))
    label_mat = rearrange_save2mat(numpy.array(label_set_list), len(label_list), save2path=os.path.join(mat_save_path, mat_name_list[1]))
    return vib_mat, label_mat


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
