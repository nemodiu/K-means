import numpy as np

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
