import numpy as np
from sklearn.datasets import make_blobs
from collections import  Counter
from sklearn import preprocessing


def bernoulli(probability):
    arr_ber = np.random.rand(1, 1)
    if arr_ber<=probability:
        return 1
    else:
        return 0


def euclidean_distance(a,b):
    sum=0
    for x in range(a.shape[0]):

        sum+=(a[x]-b[x])**2

    return sum


def perturb(list,privacy_cost):

    unperturb_list=list.reshape(-1,1)
    probility = np.exp(privacy_cost) / (np.exp(privacy_cost) + 1)
    print(probility)
    perturb_list = []
    for x in unperturb_list:
        if bernoulli(probility):
            perturb_list.append(x)
        else:
            perturb_list.append(1-x)
    return np.array(perturb_list).reshape(list.shape)


def aggregeate(unaggregeate_list,privacy_cost):
    probility = 2 / (np.exp(privacy_cost) + 1)
    print("******************f=",probility)
    list=np.sum(unaggregeate_list,axis=0)
    num=unaggregeate_list.shape[0]
    aggregeate_list=[]
    for x in list:
        a=(x-(0.5*probility*num))/(1-probility)
        aggregeate_list.append(a)
    return aggregeate_list


# def group(true_list,centroids_list):
#
#     swap_k_list=[]
#     for x in range(centroids_list.shape[0]):
#         swap_k_list.append([])
#
#     for x in true_list:
#         upper=1000
#         k=0
#         for i,xx in enumerate(centroids_list):
#             distance=euclidean_distance(x,xx)
#             if distance< upper:
#                 upper = distance
#                 k = i
#         swap_k_list[k].append(x)
#
#     return swap_k_list

def group1(true_list,centroids_list,pertutb_list):

    swap_k_list=[]
    for x in true_list:
        upper=1000
        k=0
        for i,xx in enumerate(centroids_list):
            distance=euclidean_distance(x,xx)
            if distance< upper:
                upper = distance
                k = i
        swap_k_list.append(k)
    print(Counter(swap_k_list))


    group_list = []
    for x in range(centroids_list.shape[0]):
        group_list.append([])
    for x in range(len(swap_k_list)):
        group_list[swap_k_list[x]].append(pertutb_list[x])


    return group_list






Epsilon=0.0025
K=3


# 模拟数据
# list1=np.ones((10000, 1))
# list=np.append(list1,np.zeros((90000,1))).reshape(-1,1)
# np.random.shuffle(list)
# list2=np.ones((80000, 1))
# l1=np.append(list2,np.zeros((20000,1))).reshape(-1,1)
# np.random.shuffle(l1)
# list3=np.ones((40000, 1))
# l2=np.append(list3,np.zeros((60000,1))).reshape(-1,1)
# np.random.shuffle(l2)
# test_list=np.hstack((list,l1,l2))
# print(np.sum(test_list,axis=0),test_list.shape)
# # 初始聚类中心
# cen=np.random.random((K,test_list.shape[1]))
# #扰动
# perturb_list=perturb(test_list,Epsilon)
# print(np.sum(perturb_list,axis=0),perturb_list.shape)
#
# #聚合
# result=aggregeate(perturb_list,Epsilon)
# print(result)

#2

# centers=np.array([[-.2, .7], [-.2, -.2], [.2, .2], [.7,.7]])
# centers=np.random.random((4,2))
# print(centers)
#
# method3_data, y_true = make_blobs(n_samples=5000000, n_features=2, centers=[[.5, -.5], [-.5, .5], [.0, -.5]],
#                   cluster_std=[0.06, 0.08, 0.08], random_state=9)
#
# method3_data_minmax= preprocessing.MinMaxScaler().fit_transform(method3_data)
#
# #print(method3_data_minmax.shape,method3_data_minmax[:],method3_data)
#
# perturb_list2=perturb(method3_data,Epsilon)
#
# group_result1=group1(method3_data,centers,perturb_list2)
# for x in group_result1:
#     print(len(x))


# 数据扰动

data_set3="F:\\data_gen\\dataset1\\test1_norm_01.csv"
method1_data3 = np.loadtxt(open(data_set3,"rb"),delimiter=",",skiprows=0,dtype=int)

print(method1_data3.shape,np.sum(method1_data3,axis=0))

perturb_method1_data3=perturb(method1_data3,Epsilon)

print(perturb_method1_data3.shape,np.sum(perturb_method1_data3,axis=0))

np.savetxt('F:\\data_gen\\dataset1\\perturb_p05_test1_norm_01.csv',perturb_method1_data3,delimiter=',')