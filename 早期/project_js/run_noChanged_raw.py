#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
# import time
import numpy
from scipy.io import loadmat
from scipy.signal import hilbert
from scipy.fftpack import fft
import collections
import json
import shutil
import itertools

import tensorflow as tf
from sklearn import metrics

from CNN_model import CNN
from Config import Config


import stat
def TestForChangeToWrite(path):
   # This is platform indepedent.
   if not os.access(path,os.R_OK):
       os.chmod(path,stat.S_IREAD)
   if not os.access(path,os.W_OK):
       os.chmod(path,stat.S_IWRITE)


os.environ['CUDA_VISIBLE_DEVICES'] = Config.env   # 确定使用哪块GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 程序本身相关信息打印程度

tensorboard_dir = 'tensorboard'
save_dir = 'checkpoints'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径
save_result = 'results'

# sys.setrecursionlimit(900000)

# 获取频谱
def fft_trans(array):
    N = len(array)
    return abs(numpy.fft.fft(array))[:, : int(N/2)] / N

# 获取包络谱
def enelop_trans(array):
    array_h = numpy.imag(hilbert(array))
    array_ene = abs(fft((array ** 2 + array_h ** 2) ** 0.5)) / len(array[0])
    array_ene[:, 0] = 0
    return array_ene[:, :int(len(array[0]) / 2)]


# 获取数据
def load_mat_func(path):
    '''
    传入路径，导出数据
    :param path: 文件路径
    :return: 包含某种故障类型下的6个样本集或者标签集的列表
    '''
    data = loadmat(path)
    list_data = ['data0', 'data1', 'data2', 'data3', 'data4', 'data5']
    return [data[list_data[i]] for i in range(6)]

# # 归一化,目前不需要
# def norm(data, method='min-max'):
#     if method =='min-max':
#         return (data - numpy.min(data)) / (numpy.max(data) - numpy.min(data))
#     if method == 'Z-score':
#         return (data - numpy.mean((data))) / numpy.std(data, ddof=1)
    
    # (data - numpy.mean(data, axis=1)) / numpy.std(data, axis=1, ddof=1)


# 打乱顺序
def shuffle(data, label):
    '''
    打乱下标，打乱数据
    '''
    indices = numpy.random.permutation(numpy.arange(len(label)))
    return data[indices], label[indices]


# deed dict generator
def feed_data(x_batch, y_batch, dropout_keep_prob_fc, is_training):
    '''
    generate the feed data dict object
    '''
    feed_dict = {model.input_x: x_batch,
                model.input_y: y_batch,
                model.dropout_keep_prob_fc:dropout_keep_prob_fc,
                model.is_training: is_training }
    return feed_dict


def batch_iter(data, label, batch_size):
    '''
    generating batch data, and return a iteration.
    -------------------------------------------------------------------------------
    '''
    data_len = len(label)
    num_batch = int((data_len - 1) / batch_size) + 1

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        # len(data)
        yield data[start_id:end_id], label[start_id:end_id]


def evaluate(sess, x_, y_):
    '''
    评估在某一数据上的平均准确率、平均损失和平均运行时间
    '''

    data_len = len(y_)

    total_loss = 0.0
    total_acc = 0.0
    y_pred_results = []
    batch_eval = batch_iter(x_, y_, 1)   # batch_size =1 while valuation & testing
    for x_batch, y_batch in batch_eval:
        # batch_len = len(y_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0, False)   # EVAL: keep_prob=1.0, valuation & testing are not during training.
        loss, acc, y_pred_batch = sess.run([model.loss, model.acc, model.y_pred], feed_dict=feed_dict)
        total_loss += loss 
        total_acc += acc
        y_pred_results.append(y_pred_batch.tolist()[0])
    return float(total_loss / data_len), float(total_acc / data_len), y_pred_results


def train(x_train_raw, y_train_raw, train_ratio):
    '''

    :param x_train_raw: 样本训练数据集
    :param y_train_raw: 标签训练数据集
    :param train_ratio: 训练集中训练数据划分比率
    :return:
    '''
    '''
    模型训练函数，批量训练一段时间后对模型进行验证，以此判断是否提前结束模型训练过程。
    每次调用函数，数据集先打乱顺序，后划分训练集和测试集
    '''

    x_train_raw, y_train_raw = shuffle(x_train_raw, y_train_raw)  # type(x_train) = ndArray

    split_train = int(len(y_train_raw) * train_ratio)
    
    x_train = x_train_raw[:split_train, :]
    x_valid = x_train_raw[split_train:, :]

    y_train = y_train_raw[:split_train, :]
    y_valid = y_train_raw[split_train:, :]

    print('''Configuring TensorBoard and Saver...''')
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    # 配置 Saver
    saver = tf.train.Saver()
    # 创建session
    # GPU Configuration
    gpu_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    gpu_config.gpu_options.allow_growth = True
    # create a session
    with tf.Session(config=gpu_config) as sess:
    #with tf.Session() as sess:
        tf.global_variables_initializer().run()
        writer.add_graph(sess.graph)
        print('Training and evaluating...')
        total_batch = 0  # 总批次
        best_acc_val = 0.0  # 最佳验证集准确率
        last_improved = 0  # 记录上一次提升批次
        require_improvement = Config.require_improvement  # 如果超过require_improvement轮未提升，提前结束训练

        loss_value_train = []   # loss value list while training
        loss_value_valid = []   # loss value list while validating
        acc_value_train = []   # acc value list while training
        acc_value_valid = []   # acc value list while validating
        # time_set_train = []   # time while training every epoch
        # time_set_valid = []   # average time while validating
        y_pred_list = []
        
        flag = False
        for epoch in range(Config.num_epochs):
            # print('Epoch:', epoch + 1)
            batch_train = batch_iter(x_train, y_train, Config.batch_size)

            # loss_train_epoch = []
            # acc_train_epoch = []

            for x_batch, y_batch in batch_train:
                # print(x_batch[0].shape)
                # print(y_batch.shape)
                feed_dict = feed_data(x_batch, y_batch, Config.dropout_keep_prob_fc, True)   # Training

                _, loss_train, acc_train, y_pred_batch = sess.run([model.optim, model.loss, model.acc, model.y_pred], feed_dict=feed_dict)

                loss_value_train.append(float(loss_train))
                acc_value_train.append(float(acc_train))

                y_pred_list.append(y_pred_batch)

                if total_batch % 1000 == 0:
                    print('current total batch is: ', total_batch)

                if total_batch % Config.save_per_batch == 0:
                    # 每多少轮次将训练结果写入tensorboard scalar
                    s = sess.run(merged_summary, feed_dict=feed_dict)
                    writer.add_summary(s, total_batch)

                if total_batch % Config.print_per_batch == 0:
                    # 每多少轮次输出在训练集和验证集上的性能
                    loss_val, acc_val, _ = evaluate(sess, x_valid, y_valid)

                    loss_value_valid.append(float(loss_val))
                    acc_value_valid.append(float(acc_val))

                    if acc_val > best_acc_val:
                        # 保存最好结果
                        best_acc_val = acc_val
                        last_improved = total_batch
                        print('model save in path: ', save_path)
                        saver.save(sess=sess, save_path=save_path)
                        print('Save success!')
                total_batch += 1

                if total_batch - last_improved > require_improvement:
                    # 验证集正确率长期不提升，提前结束训练
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break  # 跳出循环
            if flag:  # 同上
                print(total_batch)
                break
    
    # Results Ordered dict
    train_result = collections.OrderedDict()
    # train_result ={}
    # print('loss_value_train type and length is:', type(loss_value_train), '   ', len(loss_value_train))
    train_result['loss_value_train'] = loss_value_train
    # print(train_result['loss_value_train'])
    # print('acc_value_train type and length is:', type(acc_value_train), '   ', len(acc_value_train))
    train_result['acc_value_train'] = acc_value_train
    # print('time_set_train type and length is:', type(time_set_train), '   ', len(time_set_train))
    # train_result['time_set_train'] = time_set_train
    train_result['NA'] = '---------------------------------------------'
    # print('loss_value_valid type and length is:', type(loss_value_valid), '   ', len(loss_value_valid))
    train_result['loss_value_valid'] = loss_value_valid
    # print('acc_value_valid type and length is:', type(acc_value_valid), '   ', len(acc_value_valid))
    train_result['acc_value_valid'] = acc_value_valid
    # print('time_set_valid type and length is:', type(time_set_valid), '   ', len(time_set_valid))
    # train_result['time_set_valid'] = time_set_valid
    # Save results into a json file and a txt file
    json_file = os.path.join(save_result, 'result.json')
    # print('\n', json_file)
    # with open(json_file, 'w', encoding='utf-8') as json_file:
    with open(json_file, 'w', encoding='utf-8') as json_file:
        json.dump(train_result, json_file, indent=4)

    # try:
    # except: pass

def test(x_test, y_test):

    loss_value_test = []   # loss value list while testing
    acc_value_test = []   # acc value list while testing
    # time_set_test = []   # average time while testing

    msg_list = []   # testing results

    gpu_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    gpu_config.gpu_options.allow_growth = True

    # session = tf.Session()
    with tf.Session(config=gpu_config) as sess_test:
    #with tf.Session() as sess_test:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        print('loading model from: ', save_path)
        saver.restore(sess=sess_test, save_path=save_path)  # 读取保存的模型
        print('loaded!')

        print('Testing...')
        loss_test, acc_test, y_pred_test = evaluate(sess_test, x_test, y_test)  # todo

        test_result = collections.OrderedDict()
        test_result['acc_test'] = acc_test
        test_result['loss_test'] = loss_test
        # test_result['time_test'] = time_test
        test_result['y_pred_test'] = y_pred_test
        test_result['y_test'] = y_test.tolist()

        test_result['learning_rate'] = Config.learning_rate
        test_result['dropout_keep_prob_fc'] = Config.dropout_keep_prob_fc
        
        test_path = 'test_result' + '.json'
        json_file = os.path.join(save_result, test_path)
        with open(json_file, 'w', encoding='utf-8') as json_file:
            json.dump(test_result, json_file, indent=4)
        return acc_test, loss_test

class js_methods(object):
    # 嵌套类A.__dict,后可用json序列化
    def __init__(self, num, lr, dropout_fc, acc, loss):
        self.num = num
        self.lr = lr
        self.dropout_fc = dropout_fc
        self.acc = acc
        self.loss = loss
    def __repr__(self):
        return repr((self.num, self.lr, self.dropout_fc, self.acc, self.loss))

if __name__ == '__main__':
    json_config = os.path.join(save_result, 'config.json')

    print("Loading data...")

    norm_data_path = os.path.join(Config.PATH, 'norm_data.mat') # shape （6,1200,2048）
    norm_data = load_mat_func(norm_data_path)
    norm_label_path = os.path.join(Config.PATH, 'norm_label.mat')
    norm_label = load_mat_func(norm_label_path)

    inner600_data_path = os.path.join(Config.PATH, 'inner_600_data.mat')
    inner600_data = load_mat_func(inner600_data_path)
    inner600_label_path = os.path.join(Config.PATH, 'inner_600_label.mat')
    inner600_label = load_mat_func(inner600_label_path)

    outer600_data_path = os.path.join(Config.PATH, 'outer_600_data.mat')
    outer600_data = load_mat_func(outer600_data_path)
    outer600_label_path = os.path.join(Config.PATH, 'outer_600_label.mat')
    outer600_label = load_mat_func(outer600_label_path)

    # 列表切片获得的还是列表，列表相加表示列表中的元素都放在一个列表中,numpy.concatenate((list1+list2), axis=0),列表中的数组元素按列对齐拼接，并降一维（在列表的维数和ndarray的维数和上）返回一个数组
    x_train = numpy.concatenate((norm_data[:4] + inner600_data[:4] + outer600_data[:4]), axis=0)
    #print(x_train.shape) #(14400, 2048)
    y_train = numpy.concatenate((norm_label[:4] + inner600_label[:4] + outer600_label[:4]), axis=0)
    #print(y_train.shape) # (14400, 3)
    x_test = numpy.concatenate((norm_data[5:] + inner600_data[5:] + outer600_data[5:]), axis=0)
    y_test = numpy.concatenate((norm_label[5:] + inner600_label[5:] + outer600_label[5:]), axis=0)
    
    x_train = enelop_trans(x_train)
    x_test = enelop_trans(x_test)
    # print('X_train number is: ', len(x_train))
    # print('X_test number is: ', len(x_test))
    
    
    print('X_train shape is: ', x_train.shape)
    print('X_test shape is: ', x_test.shape)
    
    # # 归一化
    # TODO
    # for i, j in zip(range(len(x_train)), range(len(x_train[0]))):
    #     x_train[i][j] = norm(x_train[i][j], method='Z-score')
    # for i, j in zip(range(len(x_test)), range(len(x_test[0]))):
    #     x_test[i][j] = norm(x_test[i][j], method='Z-score')
    
    results = []

    # for p in itertools.product(range(Config.repeat), Config.dropout_keep_prob_list, Config.learning_rate_list):
    for (i, j, k) in itertools.product(
                                    range(Config.repeat), 
                                    Config.learning_rate_list,
                                    Config.dropout_keep_prob_fc_list):

        print('Current working space is: ', os.getcwd())
        print('Configuring and Saving CNN model...')

        for path in [tensorboard_dir, save_dir, save_result]:
            if not os.path.exists(path):
                os.makedirs(path)

        Config.learning_rate = j
        Config.dropout_keep_prob_fc = k

        config = Config()
        model = CNN(config)

        # copy Config.py to save_result

        # print(x_train.size)
        train(x_train, y_train, train_ratio=config.RATIO_train)
        acc_test, loss_test = test(x_test, y_test)

        results.append(js_methods(i, j, k, acc_test, loss_test))
        
        shutil.copy('Config.py', save_result)
        shutil.copy('CNN_model.py', save_result)
        shutil.copy('run.py', save_result)

        #TestForChangeToWrite('G:/js/code/model/tensorboard')
        #TestForChangeToWrite(save_dir)
        #TestForChangeToWrite(save_result)
        #os.makedirs('trail_' + str(i) +'_fc_dropout_' + str(k) + '_lr_' + str(j))
        #TestForChangeToWrite('trail_' + str(i) +'_fc_dropout_' + str(k) + '_lr_' + str(j))

        shutil.move(tensorboard_dir, 'trail_' + str(i) +'_fc_dropout_' + str(k) + '_lr_' + str(j))
        shutil.move(save_dir, 'trail_' + str(i) +'_fc_dropout_' + str(k) + '_lr_' + str(j))
        shutil.move(save_result, 'trail_' + str(i) +'_fc_dropout_' + str(k) + '_lr_' + str(j))

    with open('results.json', 'w') as f:
        f.write(json.dumps(results, default=lambda o: o.__dict__, indent=4))