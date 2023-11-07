import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import layers
from sklearn.preprocessing import scale
from tensorflow.python.keras.models import Model
# 讀檔

train = pd.read_csv('C:/Users/user/Downloads/NSL_GUI-20221220T055912Z-001/NSL_GUI/preKDDtrain_dnn3.csv')
test = pd.read_csv('C:/Users/user/Downloads/NSL_GUI-20221220T055912Z-001/NSL_GUI/preKDDtest_dnn2.csv')

train.drop(train.columns[[0]], axis=1, inplace=True)
test.drop(test.columns[[0]], axis=1, inplace=True)

# 將label13刪除
train = train.drop(train[train['label'] == 13].index)

# 取訓練和label

A = train.iloc[:, 0:122]
B = train.iloc[:, 122]
x_test = test.iloc[:, 0:122]
y_test = test.iloc[:, 122]


x_train = np.array(A)
x_train = scale(x_train)
y_train = np.array(B)


x_test = np.array(x_test)
x_test = scale(x_test)
y_test = np.array(y_test)

# 參數
epochs = 20
batch_size = 500
train = x_train
train_label = y_train
test = x_test
test_label = y_test

# DNN

hidden_layer = 90


class DNN(Model):
    def __init__(self):
        super(DNN, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(hidden_layer, activation='swish'),
            layers.Dropout(0.3),
            layers.Dense(13, activation='softmax')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        return encoded


model = DNN()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'],
              )

model.fit(train, train_label, epochs=epochs,
          batch_size=batch_size)

predictions = model.predict(test)

respective_accuracy = [0, 0, 0, 0, 0]
total = [0, 0, 0, 0, 0]
label_class = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
accuracy = 0
unknow = 0
unknow_T = 0
dnn_unknown = []

for i in range(len(predictions)):
    if np.max(predictions[i]) < 0.99957:
        unknow += 1
        dnn_unknown.append(i)
    if test_label[i] == 0:
        total[0] += 1
        if np.argmax(predictions[i]) == 0:
            respective_accuracy[0] += 1
            accuracy += 1
        else:
            if np.max(predictions[i]) < 0.99957:   # normal -0.8
                unknow_T += 1
    if test_label[i] >= 1 and test_label[i] < 6:
        total[1] += 1
        if np.argmax(predictions[i]) >= 1 and np.argmax(predictions[i]) < 6:
            respective_accuracy[1] += 1
            accuracy += 1
        else:
            if np.max(predictions[i]) < 0.99957:
                unknow_T += 1
    if test_label[i] >= 6 and test_label[i] < 10:
        total[2] += 1
        if np.argmax(predictions[i]) >= 6 and np.argmax(predictions[i]) < 10:
            respective_accuracy[2] += 1
            accuracy += 1
        else:
            if np.max(predictions[i]) < 0.99957:
                unknow_T += 1
    if test_label[i] >= 10 and test_label[i] < 12:
        total[3] += 1
        if np.argmax(predictions[i]) >= 10 and np.argmax(predictions[i]) < 12:
            respective_accuracy[3] += 1
            accuracy += 1
        else:
            if np.max(predictions[i]) < 0.99957:    # R2L - 0.99957
                unknow_T += 1
    if test_label[i] == 12:
        total[4] += 1
        if np.argmax(predictions[i]) == 12:
            respective_accuracy[4] += 1
            accuracy += 1
        else:
            if np.max(predictions[i]) < 0.99957:
                unknow_T += 1


print('total accuracy', accuracy/len(predictions))

for i in range(5):
    print('\n{:<6}  :{}'.format(
        label_class[i], respective_accuracy[i] / total[i]))


print(unknow)
print(unknow_T)

dnn_unknown = pd.Series(dnn_unknown)
dnn_unknown.to_csv('C:/Users/user/Downloads/NSL_GUI-20221220T055912Z-001/NSL_GUI/dnn_unknown.csv')
