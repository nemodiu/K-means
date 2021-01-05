import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from collections import  Counter
from sklearn import preprocessing
import transfor
T=10
method3_data, y_true = make_blobs(n_samples=10, n_features=2, centers=[[.5, .5], [-.5, .5], [.0, -.5]],
                  cluster_std=[0.06, 0.08, 0.08], random_state=4)

# print(method3_data)
method3_data_minmax= preprocessing.MinMaxScaler().fit_transform(method3_data)
#
# print(method3_data_minmax)
#
# np.savetxt('F:\\data_gen\\dataset1\\test1_ori.csv',method3_data,delimiter=',')
# np.savetxt('F:\\data_gen\\dataset1\\test_norm.csv',method3_data_minmax,delimiter=',')
data_set1="F:\\data_gen\\dataset1\\test_norm.csv"

method1_data = np.loadtxt(open(data_set1,"rb"),delimiter=",",skiprows=0)




#transfor.dTob(1,2)

untrans_list=method1_data.reshape(-1,1)

trans_list=[]
for x in untrans_list:

    trans_list.append(transfor.decimal_to_binary(x[0],T))

trans=np.array(trans_list)




# print(trans_list)
#
#
# cover_list=[]
# for x in trans_list:
#
#     cover_list.append(transfor.binary_to_decimal(x))
#
# print(cover_list)
# print(untrans_list)


















# x, y = method3_data_minmax[:,0], method3_data_minmax[:,1]
# ax = plt.subplot(211)  # 创建一个三维的绘图工程
# #  将数据点分成三部分画，在颜色上有区分度
# ax.scatter(x, y)  # 绘制数据点
#
#
#
# x, y = method3_data[:,0], method3_data[:,1]
# ax = plt.subplot(212)  # 创建一个三维的绘图工程
# #  将数据点分成三部分画，在颜色上有区分度
# ax.scatter(x, y)  # 绘制数据点
# plt.show()