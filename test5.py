import os
import numpy as np
from sklearn import preprocessing
import pandas as pd
from collections import Counter
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

a=[ 0 for x in range(2)]

print(a)

a=[[2.87677500e+05 ,1.02381165e+00],
 [2.67984200e+05, 1.15401191e+00],
 [2.71679862e+05 ,1.07969427e+00],
 [2.73372530e+05 ,1.12474347e+00],
 [2.46348210e+05 ,1.20428512e+00],
 [2.67655528e+05, 1.18192009e+00],
 [2.83644329e+05 ,1.11645241e+00],
 [2.52654143e+05 ,1.05824614e+00],
 [2.76291384e+05, 1.10011172e+00],
 [2.75238319e+05 ,1.16419216e+00]]

print(np.sum(np.array(a),axis=0))

a=Counter(np.array([x  for x in range(5)]))
b=dict(a)
print(b
      ,type(b))

dict1={'a':2,'e':3,'f':8,'d':4}

dict2 = sorted(dict1)

print(dict2)





st = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

print(st[:5:-1])
print(st[1::-1])


def singleNumber( nums) -> int:
 for i, x in enumerate(nums):
  flag = 0
  print(i,x)
  for y in nums[:i:-1]:
   print(y)
   if x == y:
    flag = 1
  if flag == 0:
   return x
print(singleNumber([2,2,1,1,3]))

print([2,2,1,1,3][:1
                  :-1])