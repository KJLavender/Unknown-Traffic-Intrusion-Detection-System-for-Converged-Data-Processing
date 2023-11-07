from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

test = pd.read_csv('C:/Users/user/Downloads/NSL_GUI-20221220T055912Z-001/NSL_GUI/preKDDtest_dnn2.csv')
test.drop(test.columns[[-1]], axis=1, inplace=True)
test.drop(test.columns[[0]], axis=1, inplace=True)
data = pd.read_csv('C:/Users/user/Downloads/NSL_GUI-20221220T055912Z-001/NSL_GUI/unknown.csv')
data.drop(data.columns[[0]], axis=1, inplace=True)

data = np.array(data)
k = 0
a = len(data)
arr = [0]*a


for i in range(len(test)):
    if(i in data):
        arr[k] = test.loc[i]
        k = k + 1
arr = np.array(arr)

# 標準化
# scale_arr = scale(arr)
scale_arr = MinMaxScaler(feature_range=(0, 1)).fit_transform(arr)

clustering = DBSCAN(eps=0.45, min_samples=16)
# clustering.fit_predict(scale_arr)

# 用PCA降維
pca = PCA(n_components=8)
newData = pca.fit_transform(scale_arr)
clustering.fit_predict(newData)

# 用PCA降維成2維然後用圖顯示
pca = PCA(n_components=2)
newData = pca.fit_transform(scale_arr)
newData = pd.DataFrame(newData)

x = np.array(newData.iloc[:, 0])
y = np.array(newData.iloc[:, 1])

# 在2維上畫出分類結果
plt.scatter(x, y, c=clustering.labels_)
plt.show()

# 分成幾類
a = pd.DataFrame(clustering.labels_)
print(a.value_counts())

# 上標籤
newarr = pd.DataFrame(arr)
newarr['label'] = clustering.labels_

newarr.to_csv('C:/Users/user/Downloads/NSL_GUI-20221220T055912Z-001/NSL_GUI/dbscan.csv')
