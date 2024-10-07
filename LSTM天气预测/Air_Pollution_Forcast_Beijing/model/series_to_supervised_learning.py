#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
Time:17/1/1
---------------------------
Question: LSTM data Preparation(normalizing the input variables)
          How to Convert a Time Series to a Supervised Learning Problem in Python?

---------------------------
"""
#  多变量预测
import pandas as pd


# data: 作为列表或2D Numpy数组的观察序列
# n_in: 作为输入的滞后观测值数(X); n_out: 作为输出的观测值数 （y）n_in和n_out用来调节输入时间长度和预测时间长度
# dropnan: 是否删除具有NaN值的行的布尔值, 默认为True
def series_to_supervised(data, columns, n_in=1, n_out=1, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]  # shape[1]表示第二列
    df = pd.DataFrame(data)  # 转换为带标签的二维数组
    cols, names = list(), list()  # cols, names均为list型数据
    # 输入序列(t-n, ... t-1)
    for i in range(n_in, 0, -1):  # 循环range(start, stop[, step])
        cols.append(df.shift(i))  # 序列向前移动i个时间步长(i=1), append方法: 向末尾追加元素
        names += [('%s%d(t-%d)' % (columns[j], j + 1, i)) for j in range(n_vars)]
    # 预测序列(t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))  # 序列向后移动i个时间步长
        if i == 0:
            names += [('%s%d(t)' % (columns[j], j + 1)) for j in range(n_vars)]
        else:
            names += [('%s%d(t+%d)' % (columns[j], j + 1, i)) for j in range(n_vars)]
    # 将输入和预测放在一起
    agg = pd.concat(cols, axis=1)  # 纵向连接DataFrame对象, axis=1表示按column方向拼接(横向拼接)
    agg.columns = names
    # 去掉带有NaN值的行
    if dropnan:
        clean_agg = agg.dropna()
    return clean_agg
    # 为监督学习而构建的pandas DataFrame


if __name__ == '__main__':
    values = [x for x in range(10)]  # 示例
    data = series_to_supervised(values, ['temp'], 2)
    print(data)

"""
  一步单变量预测
#from pandas import concat


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

values = [x for x in range(10)]
data = series_to_supervised(values)
print(data)

"""