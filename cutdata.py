import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from collections import  Counter
from sklearn import preprocessing
import transfor

def string_to_int(string):
    data=[]
    for x in string:
        data.append(int(x))
    return data

def int_to_string(num):
    data=[]

    for x in num:
        data.append(str(x))
    return data

data_set1="F:\\data_gen\\dataset1\\small_test1_ori.csv"
method1_data1 = np.loadtxt(open(data_set1,"rb"),delimiter=",",skiprows=0)
data_set2="F:\\data_gen\\dataset1\\small_test1_norm.csv"
method1_data2 = np.loadtxt(open(data_set2,"rb"),delimiter=",",skiprows=0)

data_set3="F:\\data_gen\\dataset1\\small_test1_norm_01.csv"
method1_data3 = np.loadtxt(open(data_set3,"rb"),delimiter=",",skiprows=0,dtype=int)


# np.savetxt('F:\\data_gen\\dataset1\\small_test1_ori.csv',method1_data[:500000],delimiter=',')
# np.savetxt('F:\\data_gen\\dataset1\\small_test1_norm.csv',method1_data2[:500000],delimiter=',')
#np.savetxt('F:\\data_gen\\dataset1\\small_small_test1_norm_01.csv',method1_data3[:50000],delimiter=',')
a=method1_data1[:10]
print(a)
b=method1_data2[:10]
print(b)

c=method1_data3[:10]


cover_list=[]
for x in c.reshape(-1,10):



    x=int_to_string(x)


    tempb=transfor.binary_to_decimal(x)

    cover_list.append(tempb)

cover_list=np.array(cover_list).reshape(a.shape)
print(cover_list)

print(c)