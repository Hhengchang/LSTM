#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
Time:17/1/1
---------------------------
Question:
---------------------------
"""
from keras import Sequential
from keras.layers import LSTM, Dense
from Air_Pollution_Forcast_Beijing.model.data_tranform import scaler, test_x, train_X, test_X, train_y, test_y
import matplotlib.pyplot as plt
from numpy import concatenate  # 数组拼接
from math import sqrt
from sklearn.metrics import mean_squared_error

model = Sequential()  # 顺序模型
# 指定input_shape,该参数需要包含时间步长数和特征数的元组
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))  # 隐藏层50个神经元, 输出层一个神经元
model.add(Dense(1))  # 全连接层, 输出维度为1
model.compile(loss='mae', optimizer='adam')  # 损失函数采用MAE，优化算法采用adam
# 输入模式矩阵train_X和匹配输出模式数组train_y, test_X, test_y用来评估损失
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y))

'''
    对数据绘图
'''
plt.plot(history.history['loss'], label='train')  # history.history属性会把之前的那些值全保存在里面, 如果有验证集的话, 也包含了验证集的这些指标变化情况
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# 为了在原始数据的维度上计算损失，需要将数据转化为原来的范围再计算损失
yHat = model.predict(test_X)  # 将test_x模型放到模型中预测
# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
'''
    这里注意的是保持拼接后的数组  列数  需要与之前的保持一致
'''
inv_yHat = concatenate((yHat, test_x[:, 1:]), axis=1)   # 数组拼接, axis=1表示对应行的数组进行拼接
inv_yHat = scaler.inverse_transform(inv_yHat)
inv_yHat = inv_yHat[:, 0]

test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_x[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)    # 将标准化的数据转化为原来的范围
inv_y = inv_y[:, 0]

rmse = sqrt(mean_squared_error(inv_yHat, inv_y))
print('Test RMSE: %.3f' % rmse)
