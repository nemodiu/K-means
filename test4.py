import numpy as np
from sklearn.cluster import KMeans
# a=np.zeros((100,1))
# print(a)
# a[3::4]+=1
# print(a)

arr=np.arange(16).reshape(16,-1)
print(arr)

arr1=np.copy(arr)
temp=arr[1]
print(temp,temp.shape)
arr1[1]=arr1[2]
print(arr1)
arr1[2]=temp
print(arr1)

a=np.array([[2,1],[2,2],[1,1]])
print(a)
a=np.sort(a,axis=0)
print(a)

a=np.random.random((10,3))
print(a)

print()

a=np.array([])
print(a.shape)


arr=np.arange(16).reshape(4,-1)
print(arr)
for x in arr:
    print(x,x.shape)

def string_to_int(string):
    data=[]
    for x in string:
        data.append(int(x))
    return data

a='0111100110'

print(string_to_int(a))

data_set3="D:\\data_gen\\dataset1\\test1_norm.csv"

method1_data3 = np.loadtxt(open(data_set3,"rb"),delimiter=",",skiprows=0)

small=method1_data3




kmeans = KMeans(n_clusters = 3)
kmeans.fit(small)
result = kmeans.labels_
c=kmeans.cluster_centers_
print(c)


print(result,result.shape)
#np.savetxt('D:\\data_gen\\dataset1\\small_small_test1_norm.csv',small,delimiter=',')