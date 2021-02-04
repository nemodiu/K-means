import numpy as np
from sklearn.cluster import KMeans
from collections import  Counter
from sklearn import metrics
# a=np.zeros((100,1))
# print(a)
# a[3::4]+=1
# print(a)

# arr=np.arange(16).reshape(16,-1)
# print(arr)
#
# arr1=np.copy(arr)
# temp=arr[1]
# print(temp,temp.shape)
# arr1[1]=arr1[2]
# print(arr1)
# arr1[2]=temp
# print(arr1)
#
# a=np.array([[2,1],[2,2],[1,1]])
# print(a)
# a=np.sort(a,axis=0)
# print(a)
#
# a=np.random.random((10,3))
# print(a)
#
# print()
#
# a=np.array([])
# print(a.shape)


# arr=np.arange(16).reshape(4,-1)
# print(arr)
# for x in arr:
#     print(x,x.shape)

def string_to_int(string):
    data=[]
    for x in string:
        data.append(int(x))
    return data

# a='0111100110'
#
# print(string_to_int(a))

# data_set3="D:\\data_gen\\dataset2\\lable_3D_norm.csv"
# method1_data3 = np.loadtxt(open(data_set3,"rb"),delimiter=",",dtype=int,skiprows=0)
# print(method1_data3.shape)

# print(Counter(method1_data3))

data_set2="D:\\data_gen\\dataset2\\3D_norm.csv"
method1_data2 = np.loadtxt(open(data_set2,"rb"),delimiter=",",skiprows=0)
print(method1_data2.shape)



kmeans = KMeans(n_clusters = 10)
kmeans.fit(method1_data2)
result = kmeans.labels_
c=kmeans.cluster_centers_
print(c)

print(Counter(result))
print(result,result.shape)

score_ch=metrics.calinski_harabasz_score(method1_data2, result)


score_db=metrics.davies_bouldin_score(method1_data2, result)
print(score_ch,score_db)

# np.savetxt('D:\\data_gen\\dataset2\\label.csv',result,delimiter=',')