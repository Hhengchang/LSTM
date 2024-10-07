#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
Time:18/1/9
---------------------------
Question:   时间序列问题，利用前几天的空气污染数据预测下一段时间的空气污染情况
            Basic Data Preparation（存在的问题：2000多条数据中 多条数据 pm2.5 为空值NA --> 补为0）
---------------------------
"""

import pandas as pd
from datetime import datetime
from Air_Pollution_Forcast_Beijing.resource.util import RAW_DATA, PROCESS_LEVEL1

pd.options.display.expand_frame_repr = False  # 不允许换行显示

# raw_data = pd.read_csv(RAW_DATA)
# head:返回前几行，默认5行
# print(raw_data.head())

# 处理时间，字符串 ---> 时间格式


def parsedate(x):
    return datetime.strptime(x, '%Y %m %d %H')  # 时间格式转化为”年 月 日 时“


# index_col: 指定索引列。
# 关注对时间处理的模块
# 设置第一列为行索引(index_col=0)，parse_dates=[['', '']]传入多列名，尝试将其解析并且拼接起来
raw_data = pd.read_csv(RAW_DATA, parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=parsedate)
# drop：删除指定列(axis=0为行，axis=1为列），这里是删除No列。inplace：True表示删除某行后原dataframe变化，False不改变原始dataframe
raw_data.drop('No', axis=1, inplace=True)
# 重新设置所有的列名
raw_data.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
# 设置索引列的名字为date
raw_data.index.name = 'date'

# fillna(0):将NAN值填充为0
raw_data['pollution'].fillna(0, inplace=True)
# 去掉前24小时的数据
raw_data = raw_data[24:]
print(raw_data.head())
# 保存数据
raw_data.to_csv(PROCESS_LEVEL1)

"""
使用index的好处：
更方便的数据查询；
使用index可以获得性能提升；
自动的数据对齐功能；
更多更强大的数据结构支持；
"""
