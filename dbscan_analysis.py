from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import scale
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

# 特徵降維判斷
# scale_arr = scale(arr)
scale_arr = MinMaxScaler(feature_range=(0, 1)).fit_transform(arr)
pca = PCA(n_components=10)
pca.fit(scale_arr)

variance = pca.explained_variance_ratio_
var = np.cumsum(np.round(variance, 3)*100)
plt.figure(figsize=(12, 6))
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.ylim(0, 100.5)
plt.plot(var)
plt.show()

# 肘型法判斷eps
plt.figure(figsize=(10, 5))
nn = NearestNeighbors(n_neighbors=10).fit(scale_arr)
distances, idx = nn.kneighbors(scale_arr)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.plot(distances)
plt.show()
