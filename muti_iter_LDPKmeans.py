import numpy as np
from collections import Counter
from sklearn.datasets import make_blobs
from sklearn import metrics
import prettytable as pt

def bernoulli(probability):
    arr_ber = np.random.rand(1, 1)
    if arr_ber<=probability:
        return 1
    else:
        return 0


def encode(unencode_list):
    encode_list=[]
    for x in unencode_list:
        # if x>1 or x<-1:
        #     print("somethings wrong")
        if bernoulli((x+1)/2):
            encode_list.append(1)
        else:
            encode_list.append(-1)
    return np.array(encode_list).reshape(unencode_list.shape)


def perturb(privacy_cost, unperturb_list):
    probility=np.exp(privacy_cost)/(np.exp(privacy_cost)+1)
    #print('probility:  ',probility)
    perturb_list=[]
    for x in unperturb_list:
        if bernoulli(probility):
            perturb_list.append(x)
        else:
            perturb_list.append(-x)
    return np.array(perturb_list).reshape(unperturb_list.shape)


def ldp_agg(privacy_cost, unaggregate_list):
    unaggregate_sum=np.sum(unaggregate_list)
    agg_result=(unaggregate_sum)*(np.exp(privacy_cost)+1)/(np.exp(privacy_cost)-1)
    return agg_result

def test(test_arr,privacy_cost):
    test_number = test_arr.shape[0]
    # print(,arr1,)
    print("原始数据：", np.sum(test_arr), np.sum(test_arr) / test_number, test_arr.shape)
    e = encode(test_arr)
    print("编码数据：", np.sum(e), np.sum(e) / test_number, e.shape)

    p = perturb(privacy_cost, e)
    print("扰动数据：", np.sum(p), np.sum(p) / test_number, p.shape)
    print("恢复数据：", ldp_agg(privacy_cost, p),
          ldp_agg(privacy_cost, p) / test_number)


def swap(swap_k,true_data,zero_list):
    #print ("This is swap function")
    perturb_true_data = np.copy(true_data)
    perturb_zero_list = np.copy(zero_list)
    # print(swap_k.shape)
    # print (perturb_zero_list.shape)
    # print (perturb_true_data.shape)

    for i ,k in enumerate(swap_k):

        k=int(k[0])
        #perturb_zero_list 是否变化
        if k>1:
            temp = true_data[i]

            perturb_true_data[i] = perturb_zero_list[
                                   (k - 2) * perturb_true_data.shape[0] * perturb_true_data.shape[1] + i *
                                   perturb_true_data.shape[1]:
                                   (k - 2) * perturb_true_data.shape[0] * perturb_true_data.shape[1] + i *
                                   perturb_true_data.shape[1] + perturb_true_data.shape[1]].T[0]
            perturb_zero_list[
            (k - 2) * perturb_true_data.shape[0] * perturb_true_data.shape[1] + i * perturb_true_data.shape[1]:
            (k - 2) * perturb_true_data.shape[0] * perturb_true_data.shape[1] + i * perturb_true_data.shape[1] +
            perturb_true_data.shape[1]] = temp.reshape(perturb_true_data.shape[1], 1)



    return (perturb_zero_list,perturb_true_data)


def swap_c(swap_k,zero_list_c,k_num):
    # print(zero_list_c.shape,"dddddddd")
    perturb_zero_list_c=np.copy(zero_list_c)


    for i, k in enumerate(swap_k):
        k = int(k[0])
        if k>1:
            temp = zero_list_c[i*k_num]
            perturb_zero_list_c[i*k_num]=perturb_zero_list_c[i*k_num+k-1]
            perturb_zero_list_c[i *k_num + k-1]=temp

    return perturb_zero_list_c


def euclidean_distance(a,b):
    sum=0
    for x in range(a.shape[0]):

        sum+=(a[x]-b[x])**2

    return sum


def group(true_list,centroids_list):
    # print(true_list)
    # print(centroids_list)
    swap_k_list=[]
    for x in true_list:
        upper=1000
        k=1
        for i,xx in enumerate(centroids_list):
            distance=euclidean_distance(x,xx)
            if distance< upper:
                upper = distance
                k = i
        swap_k_list.append(k)

    return np.array(swap_k_list).reshape(true_list.shape[0],1)+1


def initialize(Epsilon,k_num,test_data):
    test_data = test_data[:]
    # 打乱数据集
    #np.random.shuffle(test_data)
    print("数据集概况：")
    print(np.sum(test_data, axis=0))
    print(test_data.shape)

    # 生成（K-1）*n*m虚拟数据
    zero_list = np.zeros(((k_num - 1) * test_data.shape[0] * test_data.shape[1], 1))
    zero_list_c=np.zeros((k_num * test_data.shape[0], 1))
    zero_list_c[::k_num]+=1
    # print(zero_list_c,zero_list_c.shape)
    # 编码并扰动
    perturb_zero_list = perturb(Epsilon, encode(zero_list))
    perturb_zero_list_c=perturb(Epsilon,encode(zero_list_c))

    print("编码+扰动数据集")
    perturb_true_data = np.empty([0, test_data.shape[0]])
    for x in test_data.T:
        perturb_x = perturb(Epsilon, encode(x))
        perturb_true_data = np.append(perturb_true_data, [perturb_x], axis=0)

    perturb_true_data = perturb_true_data.T

    return test_data,perturb_zero_list,perturb_true_data,perturb_zero_list_c


def updatecentroid(swap_perturb_true_data, swap_perturb_zero_list,swap_perturb_zero_list_c):
    #update_centroids=[]
    swap_perturb_zero_list = swap_perturb_zero_list.reshape(k_num - 1, test_data.shape[0], test_data.shape[1])

    swap_result=[]
    for i, x in enumerate(swap_perturb_zero_list):
        temp = []
        for xx in x.T:
            # print(xx.shape)
            a = ldp_agg(Epsilon, xx)
            # print(a)
            temp.append(a)
        swap_result.append(temp)

    swap_result_first = []
    for x in swap_perturb_true_data.T:
        a = ldp_agg(Epsilon, x)
        swap_result_first.append(a)

    swap_result.insert(0, swap_result_first)

    c_list_swap = []
    swap_perturb_zero_list_c = swap_perturb_zero_list_c.reshape(-1, k_num)
    for x in swap_perturb_zero_list_c.T:
        a = ldp_agg(Epsilon, x)
        c_list_swap.append(a)

    swap_result=np.array(swap_result)
    c_list_swap=np.array(c_list_swap).reshape(-1,1)
    print(swap_result)
    print(c_list_swap)
    #for x in swap_result:

    #print(swap_result.shape, c_list_swap.shape)
    update_centroids=swap_result/c_list_swap
    print(update_centroids)
    info=0
    for i,x in enumerate( update_centroids):
        flag=0
        for xx in x:
            if xx<-1 or xx>1:
                flag=1
                info=1
        if flag==1:
            update_centroids[i]=(np.random.rand(update_centroids.shape[1],)-.5)*2

    if info==1:
        print("修正过信息：",update_centroids)
    return update_centroids


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



# 参数设置：
Epsilon=.999  # 隐私预算
k_num=4     # 聚类个数
iter_max_times=20
# 初始中心点
#test_a=np.array( [[0.5,0.3,-0.2,-0.6],[0.8,0.2,0.3,0.1],[-0.2,-0.6,-0.8,-0.1],])
#test_a=np.array([[0.5, 0.8, -0.4], [-0.2, 0.1, 0.6], [0.7, 0.8, -0.4], [-0.5,-0.1,-0.2],[-0.7,0.6,0.5]])
#test_a=np.array([[0.6, 0.2, 0.8], [0.4, 0, 0.6],[0.1, -0.6, 0.5],[-0.8,-0.2,0.3],[-0.6,-0.4,0.1]])
#test_a=np.array( [[0.5,0.1,0.7],[0.1, -0.6, 0.5],[-0.7,-0.3,0.2],])

test_a=(np.random.rand(4,2)-.5)*2
iter_centroid =test_a  # 初始中心点
print ("初始中心点：",iter_centroid)

# 提取数据集
# 方法1 本地读取
data_set1="data\\Simulation_data.csv"
data_set2="data\\test1.csv"
method1_data = np.loadtxt(open(data_set1,"rb"),delimiter=",",skiprows=0)
# 方法2 内存生成
a=[[0.8, 0.8], [0.4, 0.4],[-0.4, -0.4],[-0.8,-0.8],[0.5,0.1]]
a1=np.array([a[0]]).repeat(10000,axis=0)
a2=np.array([a[1]]).repeat(20000,axis=0)
a3=np.array([a[2]]).repeat(40000,axis=0)
a4=np.array([a[3]]).repeat(20000,axis=0)
a5=np.array([a[4]]).repeat(1,axis=0)
method2_data=np.vstack((a1,a2,a3,a4,a5))

# 方法3 函数生成
method3_data, y_true = make_blobs(n_samples=100000, n_features=2, centers=[[-.7, -.7], [-.2, -.2], [.2, .2], [.7,.7]],
                  cluster_std=[0.06, 0.08, 0.08, 0.06], random_state=9)

best_ch_score=metrics.calinski_harabasz_score(method3_data,y_true)
#best_score_sc = metrics.silhouette_score(method3_data, y_true)
best_score_db=metrics.davies_bouldin_score(method3_data, y_true)

# 数据集选择
test_data=method3_data
# 初始化
test_data, perturb_zero_list, perturb_true_data,perturb_zero_list_c=initialize(Epsilon, k_num, test_data)

# 循环迭代

score_list=[]
swap_k=group(test_data,iter_centroid)

for t in range(iter_max_times):

    print ("the",t+1,  "-th iter swap counter")
    print (Counter(swap_k.flatten()))
    swap_perturb_zero_list, swap_perturb_true_data = swap(swap_k, perturb_true_data, perturb_zero_list)
    swap_perturb_zero_list_c = swap_c(swap_k, perturb_zero_list_c, k_num)

    update_centroids=updatecentroid(swap_perturb_true_data,swap_perturb_zero_list,swap_perturb_zero_list_c)

    # 分数计算
    y_pred=group(test_data,update_centroids)
    score=measurescore(test_data,y_true,y_pred)
    score_list.append(score)
    tb = pt.PrettyTable()
    tb.field_names = ["Calinski Harabasz Score ", "davies_bouldin_score", "Mutual Information based scores ", "Adjusted Rand index", "v_measure_score"]
    tb.add_row(score)
    print(tb)


    # 排序 中止循环条件
    update_centroids = np.sort(update_centroids, axis=0)
    iter_centroid = np.sort(iter_centroid, axis=0)
    if (update_centroids==iter_centroid).all():
        break
    iter_centroid=update_centroids
    swap_k=y_pred

score_list=np.array(score_list)
print(score_list,score_list.shape)
final_score=np.array([np.max(score_list[:,0],axis=0),np.min(score_list[:,1],axis=0),np.max(score_list[:,2],axis=0),
             np.max(score_list[:,3],axis=0),np.max(score_list[:,4],axis=0)])

print(final_score,final_score.shape)

# compare score

print("Best score_ch:", best_ch_score)





print ("恢复数据:perturb_zero_list")
#print (perturb_zero_list.shape)
perturb_zero_list=perturb_zero_list.reshape(k_num-1,test_data.shape[0],test_data.shape[1])
#print (perturb_zero_list.shape)
check_result=[]
for i,x in enumerate(perturb_zero_list):
    temp = []
    #print (x.shape)
    #print("-----------The",i+2,"-th centroid-------------------- ")
    for xx in x.T:
        #print(xx.shape)
        a = ldp_agg(Epsilon, xx)
        #print(a)
        temp.append(a)
    check_result.append(temp)

print ("--------------compare result-----------------")

result_first=[]
print ("恢复数据:perturb_true_data")
for x in  perturb_true_data.T:
    a = ldp_agg(Epsilon, x)
    result_first.append(a)

# swap_result.insert(0,swap_result_first)
check_result.insert(0,result_first)
# print("swap_result:",swap_result)
print("check_result:",check_result)

print ("恢复数据:perturb_zero_list_c")
c_list=[]
perturb_zero_list_c=perturb_zero_list_c.reshape(-1,k_num)
print(perturb_zero_list_c.shape)
for x in perturb_zero_list_c.T:
    a=ldp_agg(Epsilon,x)
    c_list.append(a)
print(c_list)