
from sklearn import metrics
import numpy as np

labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 0, 1, 1, 2]

# 调整兰德系数 （Adjusted Rand index）
a=metrics.adjusted_rand_score(labels_true, labels_pred)
print(a)
# 互信息评分（Mutual Information based scores）
b=metrics.adjusted_mutual_info_score(labels_true, labels_pred)
print(b)

c=metrics.mutual_info_score(labels_true, labels_pred)
print(c)

