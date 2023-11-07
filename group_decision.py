
import pandas as pd
import numpy as np
from math import dist
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler


eps = 0.5
data = pd.read_csv('C:/Users/user/Downloads/NSL_GUI-20221220T055912Z-001/NSL_GUI/dbscan.csv')
label = data.iloc[:, -1]
data.drop(data.columns[[-1]], axis=1, inplace=True)
newData = scale_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
pca = PCA(n_components=8)
newData = pca.fit_transform(scale_data)


def w(p):
    if(p < eps):
        return 1
    else:
        return 0


def densityP(p, ne, j):
    total = 0
    for i in ne:
        if(i != j):
            total += w(p[i])/p[i]
    return total


n = len(newData)
# 初始化dis矩陣
dis = np.zeros([n, n])

for i in range(n):
    for j in range(i + 1, n):
        dis[j][i] = (dist(newData[i], newData[j]))
    print("初始化dis矩陣進度：{}/{}".format(i + 1, n))

# 複製形成完滿矩陣
i_lower = np.triu_indices(n, 0)
dis[i_lower] = dis.T[i_lower]

# 印出來看
# dis_data = pd.DataFrame(dis)
# dis_data.to_csv('D:/python/AE/dis_data.csv')

# 計算密度
arr = np.zeros([n, len(label.value_counts())])
for j in range(n):
    for i in range(len(label.value_counts())):
        neighbors = np.where(label == i)[0]
        arr[j][i] = densityP(dis[j], neighbors, j)
    print("計算密度：{}/{}".format(j + 1, n))
print(label.value_counts())

# 上新標籤
newlabel = np.zeros([n])
for i in range(n):
    newlabel[i] = np.argmax(arr[i])
newlabel = pd.Series(newlabel)
newlabel = pd.to_numeric(newlabel, downcast='integer')
print(newlabel.value_counts())


# 印出來看
# arr = pd.DataFrame(arr)
# arr.to_csv('D:/python/AE/arr_data.csv')


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

newarr = pd.DataFrame(arr)
newarr['label'] = newlabel
newarr.to_csv('C:/Users/user/Downloads/NSL_GUI-20221220T055912Z-001/NSL_GUI/group_decision.csv')
