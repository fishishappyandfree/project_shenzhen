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
tdms_path_norm_3000_0 = r"G:\js\results\new_feed_x_3000-19-13-03.tdms"
tdms_path_norm_3000_1 = r"G:\js\results\x_new_num3_feed3000-14-53-24.tdms"
tdms_path_norm_5000_0 = r"G:\js\results\new_feed_x_5000-19-18-24.tdms"
tdms_path_norm_5000_1 = r"G:\js\results\x_new_num3_feed5000-14-54-42.tdms"
tdms_path_norm_list = [tdms_path_norm_3000_0, tdms_path_norm_3000_1,
                       tdms_path_norm_5000_0, tdms_path_norm_5000_1]
tdms_path_norm_list_name = ['tdms_path_norm_3000_0', 'tdms_path_norm_3000_1',
                            'tdms_path_norm_5000_0', 'tdms_path_norm_5000_1']
# norm_label = [0, 0, 1]

# inner 0.6-0.02
tdms_path_in_600_3000_0 = r"G:\js\results\x_0.6_0.02_num4_feed3000-19-01-34.tdms"
tdms_path_in_600_3000_1 = r"G:\js\results\x_0.6_0.02_num5_feed3000-15-09-18.tdms"
tdms_path_in_600_5000_0 = r"G:\js\results\x_0.6_0.02_num4_feed5000-19-02-43.tdms"
tdms_path_in_600_5000_1 = r"G:\js\results\x_0.6_0.02_num5_feed5000-15-08-38.tdms"
tdms_path_in_600_list = [tdms_path_in_600_3000_0, tdms_path_in_600_3000_1,
                         tdms_path_in_600_5000_0, tdms_path_in_600_5000_1]
tdms_path_in_600_list_name = ['tdms_path_in_600_3000_0', 'tdms_path_in_600_3000_1',
                              'tdms_path_in_600_5000_0', 'tdms_path_in_600_5000_1']
# in_600_label = [0, 1, 0]

# outer 0.6-0.02
tdms_path_out_600_3000_0 = r"G:\js\results\x_out_0.6_0.02_num6_feed3000-15-08-44.tdms"
tdms_path_out_600_3000_1 = r"G:\js\results\x_out_deg180_0.6_0.02_num6_feed3000-19-54-43.tdms"
tdms_path_out_600_5000_0 = r"G:\js\results\x_out_0.6_0.02_num6_feed5000-15-07-52.tdms"
tdms_path_out_600_5000_1 = r"G:\js\results\x_out_deg180_0.6_0.02_num6_feed5000-19-54-00.tdms"
tdms_path_out_600_list = [tdms_path_out_600_3000_0, tdms_path_out_600_3000_1,
                          tdms_path_out_600_5000_0, tdms_path_out_600_5000_1]
tdms_path_out_600_list_name = ['tdms_path_out_600_3000_0', 'tdms_path_out_600_3000_1',
                               'tdms_path_out_600_5000_0', 'tdms_path_out_600_5000_1']
# out_600_label = [1, 0, 0]

# 结果保存文件夹，保存文件格式为.mat文件，也可以保存为其他文件,只需要修改对应的保存函数即可
#mat_save_path = r"D:\_project_bigdata_from_HuTuKang\_experiment_data_analysis\data_from_machine_tool_3\mat"

mat_save_path = r"G:\js\code\data_mat_analysis\mat"



# 传感器通道名，需要修改
key = ["/'未命名'/'cDAQ9189-1D71297Mod5/ai3'",   # feed_x_axis channel number
       "/'未命名'/'cDAQ9189-1D71297Mod4/ai1'"]   # feed_y_axis channel number

# 信号截取段
signal_split = {
    # normal
    'normal':
    [
        # new_feed_x_3000-19-13-03.tdms 100002:1500000
        [100002, 1500000],
        # x_new_num3_feed3000-14-53-24.tdms 300000:1400000
        [300002, 1400000],
        # new_feed_x_5000-19-18-24.tdms 100000:800000
        [100004, 800000],
        # x_new_num3_feed5000-14-54-42.tdms 100000:800000
        [100004, 800000]
    ],

    # inner
    'inner_600':
    [
        # x_0.6_0.02_num4_feed3000-19-01-34.tdms 280000:1450000
        [280000, 1450000],
        # x_0.6_0.02_num5_feed3000-15-09-18.tdms 150000:1250000
        [150002, 1250000],
        # x_0.6_0.02_num4_feed5000-19-02-43.tdms 45000:750000
        [45000, 750000],
        # x_0.6_0.02_num5_feed5000-15-08-38.tdms 95000:800000
        [95000, 800000]
    ],
    # outer
    "outer_600":
    [
        # x_out_0.6_0.02_num6_feed3000-15-08-44.tdms 40000:1200000
        [40002, 1200000],
        # x_out_deg180_0.6_0.02_num6_feed3000-19-54-43.tdms 60000:1220000
        [60002, 1220000],
        # x_out_0.6_0.02_num6_feed5000-15-07-52.tdms 80000:780000
        [80004, 780000],
        # x_out_deg180_0.6_0.02_num6_feed5000-19-54-00.tdms 50000:750000
        [50004, 750000]
    ]
}


# 采样频率
Fs = 25600
# 截取段长度
N = 2048
# 截取倍数
T = 300


# 读取指定工况信号区间的数据
def get_signal(tdms_path, channel_num, region,cout=6):
    '''
    读取指定工况信号区间的数据
    :param tdms_path: 文件路径
    :param channel_num: 通道名
    :param region: 截取区间
    :param cout: 区间分段个数
    :return: 一个迭代器，包含6个区间的数据
    '''
     #       （路径， key[0]="/'未命名'/'cDAQ9189-1D71297Mod5/ai3'",item[0]=235000, item[1]=295000])
    lower_num = region[0]
    upper_num = region[1]
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
    yield [tdms_data[lower_num: upper_num][i::count] for i in range(count)]


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
        vib_x = get_signal(path, key[0], signal_key[index])
        #（路径， key[0]="/'未命名'/'cDAQ9189-1D71297Mod5/ai3'",[100002:1500000],count=6)
        for item in vib_x: # signal_key[index]遍历某种故障类型下每种工况下的6段数据
            #由原加工信号获取扩充后对应维度的加工信号数据(T,N)矩阵，标签数据(T,len(label))
            array, label = signal_process_meta(item, N, T, label_list)
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
