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

T=10
method3_data, y_true = make_blobs(n_samples=10, n_features=2, centers=[[.5, .5], [-.5, .5], [.0, -.5]],
                  cluster_std=[0.06, 0.08, 0.08], random_state=4)

# print(method3_data)




#
# print(method3_data_minmax)
#
# np.savetxt('F:\\data_gen\\dataset1\\test1_ori.csv',method3_data,delimiter=',')
# np.savetxt('F:\\data_gen\\dataset1\\test_norm.csv',method3_data_minmax,delimiter=',')




data_set1="D:\\data_gen\\dataset3\\taxi.csv"
method1_data = np.loadtxt(open(data_set1,"rb"),delimiter=",",skiprows=0,dtype="int")
print(method1_data.shape)
c1=method1_data[:,:1].reshape(-1,)
c2=method1_data[:,1:].reshape(-1,)

print(Counter(c1))
print(Counter(c2))
print(np.max(method1_data,axis=0))

# method2_data=method1_data[:,2:4]
# np.savetxt('D:\\data_gen\\dataset3\\taxi.csv',method2_data,delimiter=',',fmt = '%s')



# saved_data= preprocessing.MinMaxScaler().fit_transform(method1_data)
# print(saved_data)
# np.savetxt('D:\\data_gen\\dataset3\\taxi_norm.csv',saved_data,delimiter=',')












# method1_data=np.array([x for x in range(16)]).reshape(4,-1)
# print(method1_data)
# saved_data= preprocessing.MaxAbsScaler().fit_transform(method1_data)
#
# print(saved_data.shape,saved_data)

# print(saved_data.shape)
#
# temp=saved_data[:,1:]
# print(temp.shape)

#np.savetxt('D:\\data_gen\\dataset2\\LDP\\3D_norm.csv',saved_data,delimiter=',')







#transfor.dTob(1,2)

# untrans_list=method1_data.reshape(-1,1)
#
# trans_list=[]
# for x in untrans_list:
#     temp_list=transfor.decimal_to_binary(x[0],T)
#     #print(temp_list)
#     temp_list=string_to_int(temp_list)
#
#     trans_list.extend(temp_list)
#
# trans=np.array(trans_list).reshape(method1_data.shape[0],method1_data.shape[1]*10)
#
# np.savetxt('D:\\data_gen\\dataset2\\3D_norm_01.csv',trans,delimiter=',')
#
# # print(method3_data)
# print(trans,trans.shape)



# 还原

# cover_list=[]
# for x in method1_data.reshape(-1,T):
#
#
#
#     x=int_to_string(x)
#
#     b=transfor.binary_to_decimal(x)
#
#     cover_list.append(b)
#
# cover_list=np.array(cover_list).reshape((20,-1))
# print(cover_list)



















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