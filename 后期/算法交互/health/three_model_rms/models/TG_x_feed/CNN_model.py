
# coding: utf-8
from os import environ
import tensorflow as tf
import models.TG_x_feed.Config


class CNN(object):
    '''分类，CNN模型'''

    def __init__(self, Config):
        self.config = Config
        tf.reset_default_graph()                                        # 清除每次运行时，tensorflow中不断增加的节点并重置整个defualt graph
        self.is_training = tf.placeholder(tf.bool, name='is_training')  # 判断是在训练还是测试，训练则为True ， 测试则为False
        self.dropout_keep_prob_fc = tf.placeholder(tf.float32, name='dropout_keep_prob_fc')
        self.global_step = tf.Variable(0, trainable=False)        # 用于学习率衰减，衰减计算的全局步骤。 一定不为负数。喂入一次 BACTH_SIZE 计为一次 global_step
        self.input_x = tf.placeholder(tf.float32, [None, self.config.seq_width * self.config.seq_height], name='input_x') # [None, 32*64]=[None,2048]
        self.real_label = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')     # 真实的分类标签                     # [None, 3]
        self.Deep_CNN()

    def cnn_2d_nn(self, input_, filter_num, filter_height=1, filter_width=3, filter_stride_h=1, filter_stride_w=1, initialize_name='xavier_normal', w_para=None,
                b_init=0.0, padding='SAME', a_para=[], name='conv2d'):
        """
        二维卷积运算，内部调用tf.nn.conv2d，完成 X*W+bias 对运算
        :param input_:          每次进行卷积运算时，调用本函数对输入，三次调用对输入shape分别为  [None, 32, 64, 1] / [None, 15, 31, 8] / [None, 7, 15, 16]
        :param filter_num:      每次对filter数量 = out_channels = output_dim
        :param filter_height:   filter的高度 三次分别为 [7, 3, 3]
        :param filter_width:    filter的宽度 三次分别为 [7, 3, 3]
        :param filter_stride_h: fitler的沿 height方向的步长stride 三次分别为 [1, 1, 1]
        :param filter_stride_w: fitler的沿 width 方向的步长stride 三次分别为 [1, 1, 1]
        :param initialize_name: xavier初始化 主要的目标就是使得每一层输出的方差应该尽量相等
        :param w_para:          None
        :param b_init:          0.0 后面给 b 作常数0的初始化
        :param padding:         根据调用函数确定 SAME或者 VALID
        :param a_para:
        :param name:            'conv2d'

        :return:                返回 X*W+bias
        """
        with tf.variable_scope(name):
            # tf.get_variable 获取一个已经存在的变量或者创建一个新的变量
           # print("input_.shape",input_.get_shape().as_list())                         # [None, 32, 64, 1] / [None, 15, 31, 8] / [None, 7, 15, 16]
           # print("input_.get_shape()[-1].shape",input_.get_shape()[-1])               # 1 / 8 / 16
            #print('filter_num value',filter_num)                                       # [int] 8 /16 / 32
            w_conv = tf.get_variable('weight', [filter_height, filter_width, input_.get_shape()[-1], filter_num], initializer=tf.contrib.layers.xavier_initializer())
            # input_.get_shape()[-1] 获取输入input的最后一个维度
           # print("w_conv.shape",w_conv.get_shape().as_list())                         # [7, 7, 1, 8] / [3, 3, 8, 16] / [3, 3, 16, 32]
            b_conv = tf.get_variable('biases', [filter_num], initializer=tf.constant_initializer(b_init))
           # print(" b_conv.shape",b_conv.get_shape().as_list())                        # [8] / [16] / [32]
            conv = tf.nn.conv2d(input_, w_conv, strides=[1, filter_stride_h, filter_stride_w, 1], padding=padding) + b_conv
            #print(" conv.shape",conv.get_shape().as_list())                             # [None, 32, 64, 8] / [None, 15, 31, 16] / [None, 7, 15, 32]
        return conv
    
    def full_connect(self, input_, _fc_size=1, initialize_name='xavier_normal', w_para=None, b_init=0.0, name='Full_Connection'):
        """
        全连接层  调用 nn.xw_plus_b 返回 XW+b
        :param input_:          两次调用的shape 分别为 [None, 672], [None, 64]
        :param _fc_size:        全连层的神经元个数 分别为 64(fc1层)， 3(softmax层)
        :param initialize_name: xavier初始化 主要的目标就是使得每一层输出的方差应该尽量相等
        :param w_para:
        :param b_init:
        :param name:            'Full_Connection'

        :return:                 XW+b
        """
        with tf.variable_scope(name):
          #  print("full_connect function input_.shape ",input_.get_shape().as_list())    # [None, 672] / [None, 64]
            fc_weight = tf.get_variable(name='weight', shape=[int(input_.get_shape()[-1]), _fc_size], initializer=tf.contrib.layers.xavier_initializer())
           # print(" fc_weight.shape",fc_weight.get_shape().as_list())                    # [672, 64] / [64, 3]
            fc_bias = tf.get_variable(name='bias', shape=[_fc_size], initializer=tf.constant_initializer(value=b_init))
           # print(" fc_bias.shape",fc_bias.get_shape().as_list())                        # [64] / [3]
            # output
            output_ = tf.nn.xw_plus_b(input_, fc_weight, fc_bias, name=name)             # 相当于tf.matmul(x, weights) + biases
           # print(" output_.shape",output_.get_shape().as_list())                        # [None, 64] / [None, 3]
        return output_

    # ==================================================== 前向传播 ====================================================
    def Deep_CNN(self):
        """
        卷积层-->全连接层-->softmax
        """
        #tf.variable_scope可以让变量有相同的命名，包括tf.get_variable得到的变量，还有tf.Variable的变量
        #tf.name_scope可以让变量有相同的命名，只是限于tf.Variable的变量

        #================================================= Reshape ======================================================
        with tf.variable_scope('Reshape'):
            # reshape 成[-1, 32,64,1] = [None, 32, 64, 1]
            input_x_reshape = tf.reshape(self.input_x,                                     # [None, 32, 64, 1]
                                         [-1,                                              # None
                                          self.config.seq_height,                          # 32
                                          self.config.seq_width,                           # 64
                                          1],                                              # 1
                                          name='x-reshape')

        with tf.variable_scope('CNN_Layers'):
            #=============================================== Layer_1 ===================================================
            self.cnn_0 = self.cnn_2d_nn(input_x_reshape,                                    # [None, 32, 64, 1]
                                        filter_num      = self.config.filter_num[0],        # 8
                                        filter_height   = self.config.filter_height[0],     # 7
                                        filter_width    = self.config.filter_width[0],      # 7
                                        filter_stride_h = self.config.filter_stride_h[0],   # 1
                                        filter_stride_w = self.config.filter_stride_w[0],   # 1
                                        padding         = self.config.filter_padding[0],    # 'SAME'
                                        name            = 'conv_0')
            #print("self.cnn_0.shape",self.cnn_0.get_shape().as_list())                      # [None, 32, 64, 8]
            self.activ_0 = tf.nn.elu(self.cnn_0, name='activate_0')
            #print("self.activ_0.shape",self.activ_0.get_shape().as_list())                  # [None, 32, 64, 8]
            # 最大池化
            self.pool_0 = tf.nn.max_pool(self.activ_0,
                                        ksize   = self.config.pool_ksize[0],                # [1,3,3,1]
                                        strides = self.config.pool_strides[0],              # [1, 2, 2, 1]
                                        padding = self.config.pool_padding[0],              # 'VALID'
                                        name    = 'pool_0')
            #print('self.pool_0 shape_size is: ', self.pool_0.get_shape().as_list())         # [None, 15, 31, 8]
         #====================================================== Layer_2 ===============================================
            self.cnn_1 = self.cnn_2d_nn(self.pool_0,                                        # [None, 15, 31, 8]
                                        filter_num      = self.config.filter_num[1],        # 16
                                        filter_height   = self.config.filter_height[1],     # 3
                                        filter_width    = self.config.filter_width[1],      # 3
                                        filter_stride_h = self.config.filter_stride_h[1],   # 1
                                        filter_stride_w = self.config.filter_stride_w[1],   # 1
                                        padding         = self.config.filter_padding[1],    # 'SAME'
                                        name            = 'conv_1')
           # print("self.cnn_1.shape",self.cnn_1.get_shape().as_list())                      # [None, 15, 31, 16]
            self.activ_1 = tf.nn.elu(self.cnn_1, name='activate_1')
          #  print("self.activ_1.shape",self.activ_1.get_shape().as_list())                  # [None, 15, 31, 16]
            # 最大池化
            self.pool_1 = tf.nn.max_pool(self.activ_1,                                      # [None, 15, 31, 16]
                                        ksize   = self.config.pool_ksize[1],                # [1, 3, 3, 1]
                                        strides = self.config.pool_strides[1],              # [1, 2, 2, 1]
                                        padding = self.config.pool_padding[1],              # 'VALID'
                                        name    = 'pool_1')
           # print('self.pool_1 shape_size is: ', self.pool_1.get_shape().as_list())         # [None, 7, 15, 16]

            #=================================================== Layer_3 ===============================================
            self.cnn_2 = self.cnn_2d_nn(self.pool_1,                                        # [None, 7, 15, 16]
                                        filter_num      = self.config.filter_num[2],        # 32
                                        filter_height   = self.config.filter_height[2],     # 3
                                        filter_width    = self.config.filter_width[2],      # 3
                                        filter_stride_h = self.config.filter_stride_h[2],   # 1
                                        filter_stride_w = self.config.filter_stride_w[2],   # 1
                                        padding         = self.config.filter_padding[2],    # 'SAME'
                                        name            = 'conv_2')
           # print("self.cnn_2.shape",self.cnn_2.get_shape().as_list())                      # [None, 7, 15, 32]
            self.activ_2 = tf.nn.elu(self.cnn_2, name='activate_2')
            #print("self.activ_2.shape",self.activ_2.get_shape().as_list())                  # [None, 7, 15, 32]
            # 均值池化
            self.pool_2 = tf.nn.avg_pool(self.activ_2,
                                        ksize   = self.config.pool_ksize[2],                # [1, 3, 3, 1]
                                        strides = self.config.pool_strides[2],              # [1, 2, 2, 1]
                                        padding = self.config.pool_padding[2],              # 'VALID',
                                        name    = 'pool_2')
            #print('self.pool_2 shape_size is: ', self.pool_2.get_shape().as_list())         # [None, 3, 7, 32]
        #=================================================== Flatten ===================================================
        with tf.variable_scope('Flatten'):
            pool_shape = self.pool_2.get_shape().as_list()                                  # [None, 3, 7, 32]
            nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]                           # Flatten 3*7*32 = 672
            self.cnn_reshape = tf.reshape(self.pool_2, [-1, nodes], name='cnn_reshape')
           # print('self.cnn_reshape shape_size is: ', self.cnn_reshape.get_shape().as_list())  # [None, 672]
        #================================================= Full connect 0 ================================================
        with tf.variable_scope("Full_Connect"):                                                      # 稠密层（分类层）
            self.fc = self.full_connect(self.cnn_reshape, _fc_size=self.config.fc_size, name='fc') # [None, 64]
            self.fc_activate = tf.nn.elu(self.fc, name='activate_fc')                              # [None, 64]
            self.fc_dropout = tf.nn.dropout(self.fc_activate, keep_prob=self.dropout_keep_prob_fc)
           # print('fc_layer is: ', self.fc.get_shape().as_list())                             # [None, 64]
         #================================================= Softmax =====================================================
        with tf.variable_scope("Softmax"):                                                      # 预测 pred_label
            self.logits = self.full_connect(input_=self.fc_dropout, _fc_size=self.config.num_classes, name='logit')
          #  print('self.logits.shape is: ', self.logits.get_shape().as_list())     # [None, 3] 把某个概率p从[0,1]映射到[-inf,+inf]
                                                                                   # logit = log(P/(1-P))
            self.pred_label = tf.nn.softmax(self.logits, name='predict')           # 预测的标签
            #print('self.y_pred shape is: ', self.pred_label.get_shape().as_list())     # (8640, 3), [None, 3] 把一个系列数从[-inf, +inf] 映射到[0,1]，除此之外，它还把所有参与映射的值累计之和等于1

    # ==================================================== 反向传播 ====================================================

        with tf.name_scope('learning_rate'):
            # 学习率指数衰减
            """
            decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)

            Parameters:

            learning_rate : 初始学习率
            global_step   : 用于衰减计算的全局步骤。 一定不为负数。喂入一次 BACTH_SIZE 计为一次 global_step
            decay_steps   : 衰减速度，一定不能为负数，每间隔decay_steps次更新一次learning_rate值
            decay_rate    : 衰减系数，衰减速率，其具体意义参看函数计算方程(对应α^t中的α)。
            staircase     : 若 ‘ True ’ ，则学习率衰减呈 ‘ 离散间隔 ’ （discrete intervals），具体地讲，`global_step / decay_steps`是整数除法，
                            衰减学习率（ the decayed learning rate ）遵循阶梯函数；若为 ’ False ‘ ，则更新学习率的值是一个连续的过程，每步都会更新学习率。

            Returns       : 与初始学习率 ‘ learning_rate ’ 相同的标量 ’ Tensor ‘ 。
            """
            self.decayed_learning_rate = tf.train.exponential_decay(
                                                                    self.config.learning_rate,       # 0.01
                                                                    self.global_step,                # 0
                                                                    self.config.decay_steps,         # 100
                                                                    self.config.learning_rate_decay, # 0.5
                                                                    staircase=False)

        with tf.name_scope('loss_function'):
            # 防止异常的处理
            try:
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                                                                                     logits=self.logits,    # 预测标签 # log(P/(1-P)) 把某个概率p从[0,1]映射到[-inf,+inf]
                                                                                     labels=self.real_label),  # 真实标签
                                                                                     name='cross_entropy_loss')
            except:
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                                                                  logits=self.logits,
                                                                                  labels=self.real_label),
                                                                                  name='cross_entropy_loss')
            tf.summary.scalar('cross_entropy_loss', self.loss)
            # 在画loss,accuary时会用到tf.summary.scalar这个函数。

        with tf.name_scope('AdamOptimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optim = tf.train.AdamOptimizer(self.decayed_learning_rate).minimize(self.loss, self.global_step)

        with tf.name_scope('Accuracy'):
            correct_pred = tf.equal(
                tf.argmax(self.real_label, axis=1), tf.argmax(self.pred_label, axis=1))     # 预测的标签 与 真实标签 的argmax
           # print("correct_pred",correct_pred)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('accuracy', self.acc)
            # # 在画loss,accuary时会用到tf.summary.scalar这个函数。
            """
            tf.argmax(input,axis)根据axis取值的不同返回每行或者每列最大值的索引
            tf.equal逐个元素进行判断，如果相等就是True，不相等，就是False
            tf.cast 此函数是类型转换函数 将True转化为1， false 转为0
            reduce_mean()就是按照某个维度求平均值, 即求平均有几个元素是相同的   （相同的个数）/(总共的个数) = acc
            """
