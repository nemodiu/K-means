import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics
def euclidean_distance(a,b):
    sum=0
    for x in range(a.shape[0]):

        sum+=(a[x]-b[x])**2
    return sum
a=np.zeros((2,4))+4
b=np.ones((2,4))



a=np.array([0.5,0.3,-0.2,-0.6]).reshape(1,4)
a=np.repeat(a, 1,axis=0)
b=np.array([0.8,0.2,0.3,0.1]).reshape(1,4)
b=np.repeat(b, 1,axis=0)
c=np.array([-0.2,-0.6,-0.8,-0.1]).reshape(1,4)
c=np.repeat(c, 1,axis=0)

data_to_save1=np.vstack([a,b,c])
print(data_to_save1,data_to_save1.shape)

np.random.shuffle(data_to_save1)

print(data_to_save1,data_to_save1.shape)
#np.savetxt('F:\\data\\test1.csv',data_to_save1,delimiter=',')
X, y = make_blobs(n_samples=100000, n_features=2, centers=[[-.7, -.7], [-.2, -.2], [.2, .2], [.7,.7]],
                  cluster_std=[0.08, 0.08, 0.08, 0.06], random_state=9)

print(X,X.shape)
print(y,y.shape)

score=metrics.calinski_harabasz_score(X, y)
print(score)

plt.scatter(X[:, 0], X[:, 1], marker='.')  # 假设暂不知道y类别，不设置c=y，使用kmeans聚类
plt.show()




centroid=np.array([[-1, -1], [0, 0], [1, 1], [3, 3]])
y_pred = KMeans(n_clusters=4,max_iter=10,n_init=10).fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
print(y_pred,y_pred.shape)

score=metrics.calinski_harabasz_score(X, y_pred)
print(score)
plt.show()