#!usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
from numpy import ndarray

from Air_Pollution_Forcast_Beijing.resource.util import PROCESS_LEVEL1
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from Air_Pollution_Forcast_Beijing.model.series_to_supervised_learning import series_to_supervised
pd.options.display.expand_frame_repr = False  # 不允许换行显示

dataset = pd.read_csv(PROCESS_LEVEL1, header=0, index_col=0)  # header=0:设置第一行为表头，默认为第一行
dataset_columns = dataset.columns  # 返回dataset的列标签
values = dataset.values  # 返回DataFrame的Numpy表示
print(dataset)

encoder = LabelEncoder()  # 标准化标签
values[:, 4] = encoder.fit_transform(values[:, 4])  # 将各种标签分配一个可数的连续编号
values = values.astype('float32')  # 将对象转换为指定的type

scaler = MinMaxScaler(feature_range=(0, 1))  # 实现归一化
scaled = scaler.fit_transform(values)  # 先拟合数据，然后转化它将其转化为标准形式

# 将序列数据转化为监督学习数据
# n_hours = 3
# n_features = 8
reframed = series_to_supervised(scaled, dataset_columns, 1, 1)
print(reframed.head())
reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)  # 去掉不去预测的列(这里只预测污染数据)
print(reframed.head())

values = reframed.values  # 返回reframed的所有值
n_train_hours = 365 * 24
# 分为测试集和训练集
train = values[:n_train_hours, :]  # 第一年(365*24)的数据用于训练
test = values[n_train_hours:, :]  # 后面的数据用于测试

# 监督学习结果划分,test_x.shape = (, 8)
train_x, train_y = train[:, :-1], train[:, -1]
test_x, test_y = test[:, :-1], test[:, -1]
print(train_x.shape)
print(test_x.shape[0])
print(test_x.shape[1])

# 在网络的第一层必须定义预期输入数，输入必须是三维的
# 为了在LSTM中应用该数据, 需要将其格式转化为3D format, 即[Samples(行), Timesteps(特征的过去观察值), features(特征数)]
train_X = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))  # 让2D数据中的列通过一个时间步成为特征
test_X = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
