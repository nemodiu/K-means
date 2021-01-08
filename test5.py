import os
import numpy as np
from sklearn import preprocessing
import pandas as pd

import random

# data = pd.read_csv('F:\\taxi.csv')
#
# print(data,type(data))
#
# values=np.array([x for x in range(0,16)]).reshape(-1,4)
# print(values,np.sum(values))
# data = preprocessing.scale(values)
# print(data,np.sum(data))
#
# t= preprocessing.MinMaxScaler().fit_transform(values)
# t2=preprocessing.MaxAbsScaler().fit_transform(values)
# print(t,t2)
# MinMaxScaler(feature_range=(0, 1),copy=True)：将数据在缩放在固定区间的类，
# MaxAbsScaler(copy=True)：默认缩放到区间 [-1, 1].


x=0.5

print(x**10)

a=np.random.random((2))
print(a,a.shape)
b=a.tolist()
print(type(b))

