import numpy as np
from collections import Counter


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
    print ("This is swap function")
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

    return np.array(swap_k_list).reshape(true_list.shape[0],1)


#参数设置：

Epsilon=4.#隐私预算

k_num=3   #聚类个数

#提取数据集
data_set1="F:\\data\\Simulation_data.csv"
data_set2="F:\\data\\test1.csv"
test_data = np.loadtxt(open(data_set2,"rb"),delimiter=",",skiprows=0)

test_data=test_data[:]
#打乱数据集
np.random.shuffle(test_data)
print("数据集概况：")
print(np.sum(test_data,axis=0))
print(test_data.shape)

#生成（K-1）*n*m虚拟数据
zero_list=np.zeros(((k_num-1)*test_data.shape[0]*test_data.shape[1],1))

#编码并扰动
perturb_zero_list=perturb(Epsilon,encode(zero_list))


print("扰动数据集")
perturb_true_data=np.empty([0,test_data.shape[0]])
for x in test_data.T:
    perturb_x = perturb(Epsilon, encode(x))
    perturb_true_data=np.append(perturb_true_data,[perturb_x],axis=0)

perturb_true_data=perturb_true_data.T

#计算用户离中心点最近的下标
print ("------------计算用户离中心点最近的下标---------------")
test_a=np.array( [[0.5,0.3,-0.2,-0.6],
 [0.8,0.2,0.3,0.1],
 [-0.2,-0.6,-0.8,-0.1],
 ])
centroids=np.ones((k_num,test_data.shape[1]))
swap_k=group(test_data,test_a)+1

print(swap_k.shape)
print(Counter(swap_k.flatten()))

# swap_k=np.ones((test_data.shape[0],1))+3

#交换报告
print ("------------交换报告---------------")
swap_perturb_zero_list,swap_perturb_true_data=swap(swap_k,perturb_true_data,perturb_zero_list)

print("--------------验证数据--------------")

print ("恢复数据:swap_perturb_zero_list")
#print (swap_perturb_zero_list.shape)
swap_perturb_zero_list=swap_perturb_zero_list.reshape(k_num-1,test_data.shape[0],test_data.shape[1])
#print (swap_perturb_zero_list.shape)
swap_result=[]
for i,x in enumerate(swap_perturb_zero_list):
    #print (x.shape)
    #print("-----------The",i+2,"-th centroid-------------------- ")
    temp=[]
    for xx in x.T:
        #print(xx.shape)
        a = ldp_agg(Epsilon, xx)
        #print(a)
        temp.append(a)
    swap_result.append(temp)

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

print ("恢复数据:swap_perturb_true_data")

swap_result_first=[]
for x in swap_perturb_true_data.T:
    a=ldp_agg(Epsilon,x)
    swap_result_first.append(a)
result_first=[]
print ("恢复数据:perturb_true_data")
for x in  perturb_true_data.T:
    a = ldp_agg(Epsilon, x)
    result_first.append(a)
swap_result.insert(0,swap_result_first)
check_result.insert(0,result_first)
print("swap_result:",swap_result)
print("check_result:",check_result)




















