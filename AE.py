import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model
from sklearn import preprocessing
import matplotlib.pyplot as plt

# 讀檔
train = pd.read_csv('C:/Users/user/Downloads/NSL_GUI-20221220T055912Z-001/NSL_GUI/preKDDtrain_dnn3.csv')
test = pd.read_csv('C:/Users/user/Downloads/NSL_GUI-20221220T055912Z-001/NSL_GUI/preKDDtest_dnn2.csv')

# 把label和第1列的數據編號去除

test.drop(test.columns[[0]], axis=1, inplace=True)
test.drop('label', axis=1, inplace=True)
train.drop(train.columns[[0]], axis=1, inplace=True)
train.drop('label', axis=1, inplace=True)


# 轉成陣列儲存
test = tf.convert_to_tensor(test)
train = tf.convert_to_tensor(train)

# min_max_scaler = preprocessing.MinMaxScaler()
# X_train = min_max_scaler.fit_transform(df_train20)
# 標準化，維持稀疏矩陣中為0的項目

X_test = preprocessing.scale(test)
train = preprocessing.scale(train)


# AE


class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(20, activation="relu")
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(122, activation="relu")
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = AnomalyDetector()
autoencoder.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

autoencoder.fit(train, train,                     # 輸入和輸出都是x
                epochs=50,
                batch_size=500,
                shuffle=True)
autoencoder.save('D:/python/AE/AE_MODEL')
# 計算輸入和輸出的誤差
reconstructions = autoencoder.predict(X_test)
test_loss = tf.keras.losses.mae(reconstructions, X_test)


plt.hist(test_loss[None, :], bins=100)
plt.xlabel("Test loss")
plt.ylabel("Num of examples")
plt.show()

# 平均loss+1個標準差

threshold = np.mean(test_loss) + np.std(test_loss)
print(threshold)

out = 0
ae_unknown = []
for i in range(len(reconstructions)):
    if test_loss[i] >= threshold:
        out += 1
        ae_unknown.append(i)
print('Accuracy : ', (len(reconstructions) - out) / len(reconstructions))

ae_unknown = pd.Series(ae_unknown)
ae_unknown.to_csv('C:/Users/user/Downloads/NSL_GUI-20221220T055912Z-001/NSL_GUI/ae_unknown.csv')
