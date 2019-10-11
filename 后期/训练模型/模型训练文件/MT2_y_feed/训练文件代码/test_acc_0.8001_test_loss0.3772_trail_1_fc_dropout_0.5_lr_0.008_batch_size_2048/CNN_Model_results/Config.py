#!/usr/bin/python
# -*- coding: utf-8 -*-

# tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
# input_shape  = [batch_size, in_height, in_width, in_channels]
#              = [训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
# filter_shape = [filetr_Height, filter_width, in_channels(depth)，num_filters(out_channels)]
#              = [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]

# input_shape[3] = filter_shape[2] = in_channels = 图像通道数 = input.get_shape()[-1](获取input最后一个维度的大小)

# strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
# 结果返回一个Tensor，这个输出，就是我们常说的feature map

"""
在模型中出现的几个迭代相近词辨析：

repeat     ：单纯的整个模型重复跑几遍,
             每次跑模型的时候 采用不同的学习率(learning_rate)、神经元保留率(dropout_keep_prob_fc)来排列组合
             从而测试每种学习率和 保留率不同组合的效果
epoch      ： 一个完整的数据集通过网络并返回一次
iteration  ： 所有的 batch 完成一个epoch的次数
{repeat:
       {epoch:
             {iteration:
                           }
                              }
                               }

例子： 样本数 1000， 完整训练一次， epoch = 1， 若batch_size = 10， 则 iteation = 1000/10 = 100
"""

class Config(object):
    '''
    data must be a list
    label should be a numpy.ndarray
    '''
    repeat = 3                            # 不同于epoch 是epoch的再上一层循环，即整个程序重复跑 repeat 轮
    env = '0'

    ''' 对 batch_size 也可以进行循环 找出最佳的 batch_size'''
    #batch_size_list = [64, 128, 256, 512, 1024]
    batch_size_list = [64, 128, 256, 512, 1024, 2048]

    batch_size = 64  # 每批训练大小

    # num_epochs = 300                       # 总迭代轮次
    num_epochs = 1000

    require_improvement = 20              # 如果超过require_improvement轮未提升，提前结束训练

    display_epoch = 10                    # 每10个epoch在屏幕打印一次
    valid_epoch  = 1                     # 每10个epoch验证一次

    save_epoch = 1                        # 保存模型的频率



    # 经过验证 最佳学习率在 0.005-0.01之间  最高accuracy 在 96.2-97.33% 之间
    # learning_rate_list = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    # learning_rate_list = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.05,0.01,0.5,0.1]
    #learning_rate_list = [0.005, 0.008, 0.01, 0.015]    # 用于在每一个repeat，循环测试 learning_rate

    learning_rate_list = [0.005, 0.008, 0.01]
    learning_rate = 0.01                  # 用于学习率指数衰减所用 初始学习率  decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    learning_rate_decay = 0.5             # 学习率指数衰减率
    decay_steps = 100

    dropout_keep_prob_fc_list = [0.5, 1]  # # 用于在每一个repeat，循环测试 learning_rate
    dropout_keep_prob_fc = 0.5            # 神经元保留比例

    seq_height = 32                       # 把一维序列整成二维 (seq_height,seq_width)=(32,64)=2048
    seq_width = 64

    num_classes = 3                       # Softmax 的类别数
    fc_size = 64                          # 全连接层fc1大小


    filter_height = [7, 3, 3]             # 每个filter对应的height
    filter_width  = [7, 3, 3]             # 每个filter对应的width

    filter_stride_h = [1, 1, 1]
    filter_stride_w = [1, 1, 1]           # 每个filter的stride_w（不是pool的）
    filter_num = [8, 16, 32]              # 每次卷积的filter数量 = out_channels = output_dim

    filter_padding = ['SAME','SAME','SAME']    # 第一次卷积的卷积核filter的padding方式

    pool_ksize = [[1, 3, 3, 1],                  # 每一层池化层的 池化窗口size
                  [1, 3, 3, 1],
                  [1, 3, 3, 1]]

    pool_strides = [[1, 2, 2, 1],                # 每一层池化层的 池化窗口移动的步长strides
                    [1, 2, 2, 1],
                    [1, 2, 2, 1]]

    pool_padding = ['VALID','VALID','VALID']     # 每一层池化层的池化方式




    optic_name = 'NAG'
    momentum = 0.9

    cnn_init_name = 'xavier_normal'       # xavier初始化 主要的目标就是使得每一层输出的方差应该尽量相等
    fc_init_name = 'xavier_normal'

    train_ratio = 0.8
    RATIO_train = 0.75                    # 原始的训练集再划分成 训练集和验证集，训练集占比为 0.75 即 训练集：验证集： 测试集 = 60% ：20% ：20%

    #PATH = '../../../../mat/'
    #PATH = 'D:\_project_bigdata_from_HuTuKang\_experiment_data_analysis\data_from_machine_tool_3\mat'
    #PATH = '2000_4000_mat_type_data'
    PATH = 'MT2_y_feed_dataset'


    # PATH = 'mat'
    # PATH = '/media/hust/343a6bb8-665f-4ec9-8d6d-831906929903/进给轴/model/mat'
