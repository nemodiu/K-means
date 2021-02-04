import numpy as np
from sklearn.datasets import make_blobs
from collections import  Counter
from sklearn import preprocessing
from sklearn import metrics
import random
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
    #print(probility,2 / (np.exp(privacy_cost/2) + 1))
    #print("******************f=",probility)
    list=np.sum(unaggregeate_list,axis=0)
    num=unaggregeate_list.shape[0]
    #print("num:",num)
    aggregeate_list=[]
    for x in list:
        a=(x-(0.5*probility*num))/(1-probility)
        if a<0:
            a=0
        if a>num:
            a=num
        #print(a)
        aggregeate_list.append(a)
    aggregeate_list=np.array(aggregeate_list)/unaggregeate_list.shape[0]

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
    group_true_list=[]
    for x in range(centroids_list.shape[0]):
        group_list.append([])
        group_true_list.append([])
    for x in range(len(swap_k_list)):
        group_list[swap_k_list[x]].append(pertutb_list[x])
        group_true_list[swap_k_list[x]].append(true_list[x])


    return group_list,np.array(swap_k_list),group_true_list

def compute_norm(norm,T):
    result=0
    for x in range(T):
        result=result+norm[x]*(0.5**(x+1))

    return result

def measurescore(test_data, y_true,y_pred):
    print(y_pred.shape)
    y_pred=y_pred.reshape(-1,)
    # 无label_true:
    # 1.CH分数 Calinski Harabasz Score, 取值越大越好
    score_ch=metrics.calinski_harabasz_score(test_data, y_pred)

    # 2.轮廓系数（Silhouette Coefficient, 取值-1, 1之间 取值越大越好
   # score_sc = metrics.silhouette_score(test_data, y_pred)
    # 戴维森堡丁指数(DBI)——davies_bouldin_score, 取值越小越好
    score_db=metrics.davies_bouldin_score(test_data, y_pred)


    # label_true:
    # 1.Mutual Information based scores 互信息 [0,1] 取值越大越好
    score_mi=metrics.adjusted_mutual_info_score(y_true,y_pred)
    # 调整兰德系数 （Adjusted Rand index） [-1,1]取值越大越好
    score_adi= metrics.adjusted_rand_score(y_true, y_pred)
    # v_measure_score homogeneity+completeness [0,1] 取值越大越好
    score_vm=metrics.v_measure_score(y_true,y_pred)

    result_score=[score_ch,score_db,score_mi,score_adi,score_vm]

    return result_score


T=10
Epsilon=0.01667
K=10
Time=20
m=3

# 扰动数据
data_set3="D:\\data_gen\\dataset2\\perturb_2_3D_norm_01.csv"
method1_data3 = np.loadtxt(open(data_set3,"rb"),delimiter=",",skiprows=0,dtype=int)
# 原始数据
data_set2="D:\\data_gen\\dataset2\\3D_norm.csv"
method1_data2 = np.loadtxt(open(data_set2,"rb"),delimiter=",",skiprows=0)
# 标签
data_set1="D:\\data_gen\\dataset2\\label.csv"
method1_data1 = np.loadtxt(open(data_set1,"rb"),delimiter=",",skiprows=0,dtype=int)

print(method1_data3.shape,np.sum(method1_data3,axis=0))

print(method1_data2.shape,method1_data1.shape)


# centroids=[[0.24921172, 0.77609598],
#  [0.54118091, 0.21936071],
#  [0.24921172, 0.77609598]]
# cen=np.array(centroids)

cen=np.random.random((K,m))
print(cen)




# centroids0=[
#  [0.84393223 ,0.5978422  ,0.12718662],
#  [0.14040824, 0.26979882, 0.19129719],
#  [0.33183508 ,0.13331432, 0.19974007],
#  [0.53883746, 0.20067335 ,0.42218252],
#  [0.60406409, 0.39395046, 0.15053986],
#  [0.74744518 ,0.84009436 ,0.11951312],
#  [0.60129929, 0.72145621, 0.23237307],
#  [0.41503543 ,0.47182062, 0.14544407],
#  [0.69137793, 0.65182481, 0.47711987],
#  [0.63040795 ,0.12367927 ,0.18753085]]
#
#
# cen=np.array(centroids0)

print("真实标签",Counter(method1_data1))
flag=0
for i in range(Time):

    group_list,swap_k_list,group_true_list=group1(method1_data2,cen,method1_data3)
    score=measurescore(method1_data2,method1_data1 ,swap_k_list)
    print("**********score:",score)
    if flag==score[0]:
        break


    true_cen=[]
    for x in group_true_list:
        if not x:
            true_cen.append([0 for x in range(m)])

            print("warning__0")


        else:
            x=np.array(x)
            true_cen.append(np.sum(x,axis=0)/x.shape[0])

    print("true cen:",np.array(true_cen))


    flag=score[0]
    # 计算new 中心点
    new_cen=[]
    for x in group_list:
        if not x:
            new_cen.append(np.random.random(m))

            print("warning__1")

            print(1)
        else:
            x = np.array(x)
            # print(x.shape)

            agg_list = aggregeate(x, Epsilon).reshape(-1, T)
            # print(agg_list)
            temp = []
            for xx in agg_list:
                xx_result = compute_norm(xx, T)

                if xx_result>1 or xx_result<0:
                    temp.append(random.random())
                    print("warning__2 is :",xx_result)

                else:
                    temp.append(xx_result)
            print(temp,"cen:")



                # print(xx,xx.shape)lable_3D_norm
            new_cen.append(temp)

    cen = np.array(new_cen)

    #print(new_cen)
















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
# l=[0.00666,	0.01667	,0.033	,0.05,	0.0666]
# data_set3="D:\\data_gen\\dataset2\\3D_norm_01.csv"
# method1_data3 = np.loadtxt(open(data_set3,"rb"),delimiter=",",skiprows=0,dtype=int)
# print(method1_data3.shape,np.sum(method1_data3,axis=0))
#
# for i,x in enumerate(l):
#     name="perturb_"+str(i+1)+"_3D_norm_01.csv"
#     perturb_method1_data3 = perturb(method1_data3, x)
#
#     print(perturb_method1_data3.shape, np.sum(perturb_method1_data3, axis=0))
#
#     np.savetxt('D:\\data_gen\\dataset2\\'+name, perturb_method1_data3, delimiter=',')

