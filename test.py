import numpy as np




# 创建2行2列取值范围为[0,1)的数组
arr_mean = np.random.rand(2,2)
# 创建2行3列，取值范围为标准正态分布的数组
arr_ZT = np.random.randn(2,3)


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
    print('probility:  ',probility)
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

#均匀数组
arr1=np.random.rand(10000,1)
# arr1=arr1/50
#全零
arr2=np.zeros((1, 10000))
#全1
arr3=np.ones((1, 10000))
#正态 要归一化才能使用
arr4= np.random.randn(1,10000)+0.5
# maxnum=arr4.max(axis=1)
# minnum=arr4.min(axis=1)
# print(maxnum,minnum)
# arr4=arr4/5
arr4[arr4>1]=0
arr4[arr4<-1]=0
#print(arr4)
#自定义
arr5=np.zeros((1, 10000))
arr5[0,:3000]+=1
privacy_cost=1.
test_arr=arr1

def test(test_arr,privacy_cost):
    test_number = test_arr.shape[0]
    # print(,arr1,)
    print("原始数据：", np.sum(test_arr), np.sum(test_arr) / test_number, test_arr.shape)
    e = encode(test_arr)
    print("编码数据：", np.sum(e), np.sum(e) / test_number, e.shape)

    p = perturb(privacy_cost, e)
    print("扰动数据：", np.sum(p), np.sum(p) / test_number, p.shape)
    print(p.shape)
    print("恢复数据：", ldp_agg(privacy_cost, p),
          ldp_agg(privacy_cost, p) / test_number)

test(test_arr,privacy_cost)