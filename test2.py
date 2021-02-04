import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics
from collections import Counter
def euclidean_distance(a,b):
    sum=0
    for x in range(a.shape[0]):

        sum+=(a[x]-b[x])**2
    return sum
a=np.zeros((2,4))+4
b=np.ones((2,4))



# a=np.array([0.5,0.3,-0.2,-0.6]).reshape(1,4)
# a=np.repeat(a, 1,axis=0)
# b=np.array([0.8,0.2,0.3,0.1]).reshape(1,4)
# b=np.repeat(b, 1,axis=0)
# c=np.array([-0.2,-0.6,-0.8,-0.1]).reshape(1,4)
# c=np.repeat(c, 1,axis=0)
#
# data_to_save1=np.vstack([a,b,c])
# print(data_to_save1,data_to_save1.shape)
#
# np.random.shuffle(data_to_save1)
#
# print(data_to_save1,data_to_save1.shape)
#np.savetxt('F:\\data\\test1.csv',data_to_save1,delimiter=',')
# method1_data, y = make_blobs(n_samples=100000, n_features=2, centers=[[-.7, -.7], [-.2, -.2], [.2, .2], [.7,.7]],
#                   cluster_std=[0.08, 0.08, 0.08, 0.06], random_state=9)
#
# print(X,X.shape)
# print(y,y.shape)

data_set1="D:\\data_gen\\dataset4\\g6_norm.csv"
method1_data = np.loadtxt(open(data_set1,"rb"),delimiter=",",skiprows=0,dtype="float")


#score=metrics.calinski_harabasz_score(X, y)


# plt.scatter(method1_data[:, 0], method1_data[:, 1], marker='.')  # 假设暂不知道y类别，不设置c=y，使用kmeans聚类
# plt.show()





estimator= KMeans(n_clusters=10,max_iter=10,n_init=10)
estimator.fit_predict(method1_data)

y_pred =estimator.labels_
cc=estimator.cluster_centers_
print(cc)

plt.scatter(method1_data[:, 0], method1_data[:, 1], c=y_pred)
print(y_pred,y_pred.shape)

print(Counter(y_pred))
score=metrics.calinski_harabasz_score(method1_data, y_pred)
print(score)
plt.show()

#np.savetxt('D:\\data_gen\\dataset4\\geo6_label.csv',y_pred,delimiter=',')