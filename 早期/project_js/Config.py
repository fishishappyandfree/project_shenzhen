#!/usr/bin/python
# -*- coding: utf-8 -*-


class Config(object):
    '''
    data must be a list
    label should be a numpy.ndarray
    '''
    repeat = 5
    env = '0'


    # learning_rate_list = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    learning_rate_list = [0.005, 0.001, 0.0005, 0.0001]
    learning_rate = 0.01  # 学习率   0.000001(mark)
    learning_rate_decay = 0.5
    decay_steps = 100

    dropout_keep_prob_fc_list = [0.5, 1]
    dropout_keep_prob_fc = 0.5

    seq_width = 32
    seq_length = 32
    seq_height = 1
    num_classes = 3
    fc_size = 64

    filter_width = [7, 3, 3]
    filter_height = [7, 3, 3]
    stride_w = [1, 1, 1]
    stride_h = [1, 1, 1]
    output_dim = [8, 16, 32]


    batch_size = 32  # 每批训练大小
    
    num_epochs = 300  # 总迭代轮次

    require_improvement = 800   # 如果超过require_improvement轮未提升，提前结束训练

    print_per_batch = 50  # 每多少轮输出一次结果 100
    save_per_batch = 50  # 每多少轮存入tensorboard

    optic_name = 'NAG'
    momentum = 0.9

    cnn_init_name = 'xavier_normal'
    fc_init_name = 'xavier_normal'

    RATIO_train = 0.8

    PATH = 'mat'
    # PATH = 'mat'
    # PATH = '/media/hust/343a6bb8-665f-4ec9-8d6d-831906929903/进给轴/model/mat'
