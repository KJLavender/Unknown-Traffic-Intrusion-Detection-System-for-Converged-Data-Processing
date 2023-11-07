# NSL-KDD：epoch = 50
# 程式合併
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model
import matplotlib.pyplot as plt
from math import dist
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import itertools


file_path = r'C:\Users\user\Downloads\NSL_GUI\\'
# 主程式


def main():

    th = 0.99957  # DNN 臨界值S

    # 第一次訓練
    AE('first')    # 輸出 ae_unknown_num.csv
    DNN('first', th)   # 輸出 dnn_unknown_num.csv
    # 顯示未知數量    # ae_unknown_num.csv 和 dnn_unknown_num.csv 合併成 unknown_num.csv
    unknown_len = Combine()
    DBSCANN()    # 輸出 dbscan.csv
    GroupDecision()    # 輸出 group_decision.csv

    # 再訓練
    while unknown_len > 2500:
        AE('retrain')    # AE 訓練集：train + group  # 輸出 ae_unknown_num.csv

        th = th * 0.9  # print(th)
        DNN('retrain', th)    # DNN 訓練集：train + group  # 輸出 dnn_unknown_num.csv
        # 顯示未知數量     # ae_unknown_num.csv 和 dnn_unknown_num.csv 合併成 unknown_num.csv
        unknown_len = Combine()
        DBSCANN()    # 輸出 dbscan.csv
        GroupDecision()    # 輸出 group_decision.csv


def AE(status):

    train_times = 1  # 訓練次數

    for i in range(train_times * 2):
        # AE 第一次訓練
        if status == 'first':
            file = 'preKDDtrain.csv' if i % 2 == 0 else 'preKDDtest.csv'    # AE 是一次訓練一次測試
            df = pd.read_csv(f'{file_path}{file}')
        # AE 再訓練
        else:   # status == retrain
            # 訓練
            if i % 2 == 0:  # AE 是一次訓練一次測試
                # 檔案存在；第一次之後訓練集都用 train_ae(= train 不斷加 group 進去)
                if os.path.isfile(f'{file_path}preKDDtrain_ae.csv'):
                    file = 'preKDDtrain_ae.csv'
                # 檔案不存在：第一次訓練集用 train
                else:
                    file = 'preKDDtrain.csv'

                # 讀取訓練集
                df = pd.read_csv(f'{file_path}{file}')
                # 刪除合併後多的一行
                if file == 'preKDDtrain_ae.csv':
                    df.drop(["Unnamed: 0"], axis=1, inplace=True)
                # 跑訓練集時：train 和 group 要合併且儲存
                group = pd.read_csv(f'{file_path}group_decision.csv')
                group.columns = df.columns
                df = pd.concat([df, group], axis=0)
                # 儲存為 train_ae(= train + group)
                df.to_csv(f'{file_path}preKDDtrain_ae.csv')
            # 測試
            else:
                file = 'preKDDtest.csv'
                df = pd.read_csv(f'{file_path}{file}')

        i = i + 1

        # 去除label和編號，轉成陣列儲存
        df.drop('label', axis=1, inplace=True)
        df.drop(df.columns[[0]], axis=1, inplace=True)
        df = tf.convert_to_tensor(df)

        # 標準化，維持稀疏矩陣中為0的項目
        X_train = preprocessing.scale(df)

        # AE
        class AnomalyDetector(Model):
            def __init__(self):
                super(AnomalyDetector, self).__init__()
                self.encoder = tf.keras.Sequential([
                    layers.Dense(20, activation="relu")
                ])

                self.decoder = tf.keras.Sequential([
                    layers.Dense(122, activation="relu")  # 輸出122種
                ])

            def call(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

        autoencoder = AnomalyDetector()
        autoencoder.compile(optimizer='adam', loss='mae')

        # 訓練模型
        autoencoder.fit(X_train, X_train,  # 輸入和輸出都是x
                        epochs=50,
                        batch_size=500,
                        shuffle=True
                        )

        # 計算輸入和輸出的誤差
        reconstructions = autoencoder.predict(X_train)
        test_loss = tf.keras.losses.mae(reconstructions, X_train)  # 存每個 loss 值

        # 平均loss+1個標準差
        # 0.002804639736032029
        threshold = np.mean(test_loss) + np.std(test_loss)
        print("threshold = ", threshold)
        print("Loss = ", np.mean(test_loss))
        print("Accuracy = ", np.sum(test_loss < threshold) /
              len(test_loss))  # AE 辨識已知

        ae_unknown = []     # 存未知編號的 list
        if file == 'preKDDtest.csv':    # 跑測試集時(= 訓練結果)
            for j in range(len(test_loss)):
                if test_loss[j] > threshold:    # AE 辨識未知，存編號
                    ae_unknown.append(j)

    # 輸出 AE 未知編號
    ae_unknown = pd.Series(ae_unknown)
    ae_unknown.to_csv(f'{file_path}ae_unknown_num.csv')


def DNN(status, th):

    # DNN 第一次訓練
    if status == 'first':
        train = pd.read_csv(f'{file_path}preKDDtrain.csv')
        test = pd.read_csv(f'{file_path}preKDDtest.csv')
        train.drop(train.columns[[0]], axis=1, inplace=True)
        test.drop(test.columns[[0]], axis=1, inplace=True)
        # 將 label 13 刪除
        train = train.drop(train[train['label'] == 13].index)
    # DNN 再訓練
    else:   # status == retrain
        # 檔案存在：第一次之後測試集都用 train_dnn(= train 不斷加 group 進去)
        if os.path.isfile(f'{file_path}preKDDtrain_dnn.csv'):
            file = 'preKDDtrain_dnn.csv'
        # 檔案不存在：第一次測試集用 train
        else:
            file = 'preKDDtrain.csv'

        train = pd.read_csv(f'{file_path}{file}')
        test = pd.read_csv(f'{file_path}preKDDtest.csv')
        group = pd.read_csv(f'{file_path}group_decision.csv')

        # DNN再訓練的 preKDDtrain_dnn：preKDDtrain.csv + group_decision.csv
        group.columns = train.columns
        temp = group
        label = []
        a1 = [0] * len(temp['label'].value_counts())
        for i in range(len(temp['label'].value_counts())):
            a = temp['label'].value_counts()
            a1[i] = a.idxmax()
            temp = temp.drop(temp[temp['label'] == a1[i]].index)

        # 4     3055
        # 1     1787
        # 0      997
        # 3      793
        # 6      490
        # 5      381
        # 10     379

        # n-0   d-3 p-7 r-10 u-11

        for i in range(len(group)):
            if (group.label[i] == a1[1]):
                label.append(10)
            elif (group.label[i] == a1[2]):
                label.append(3)
            elif (group.label[i] == a1[3]):
                label.append(7)
            elif (group.label[i] == a1[4]):
                label.append(0)
            elif (group.label[i] == a1[5]):
                label.append(7)
            else:
                label.append(13)

        group.drop(group.columns[[-1]], axis=1, inplace=True)
        group['label'] = label
        train = pd.concat([train, group], axis=0)

        train.drop(train.columns[[0]], axis=1, inplace=True)
        # 儲存為 train_dnn(= train + group)
        train.to_csv(f'{file_path}preKDDtrain_dnn.csv')
        test.drop(test.columns[[0]], axis=1, inplace=True)
        train = train.drop(train[train['label'] == 13].index)   # DNN 再訓練的前處理結束

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
    epochs = 50
    batch_size = 500
    train = x_train
    train_label = y_train
    test = x_test
    test_label = y_test
    cm_label = []
    cm_label1 = []

    for i in range(len(test_label)):

        if test_label[i] == 0:
            cm_label.append(0)
            cm_label1.append(0)

        if test_label[i] >= 1 and test_label[i] < 6:
            cm_label.append(1)
            cm_label1.append(1)

        if test_label[i] >= 6 and test_label[i] < 10:
            cm_label.append(2)
            cm_label1.append(1)

        if test_label[i] >= 10 and test_label[i] < 12:
            cm_label.append(3)
            cm_label1.append(1)

        if test_label[i] == 12:
            cm_label.append(4)
            cm_label1.append(1)

    # DNN
    hidden_layer = 90
    x = 1

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

    # 每訓練一次會*0.9    # 0.99957
    threshold = th
    x = 1   # DNN 訓練次數
    xtimeAcc = 0
    for i in range(x):
        model = DNN()
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=False),
                      metrics=['accuracy'],
                      )
        model.fit(train, train_label, epochs=epochs,
                  batch_size=batch_size)

        # 各別準確率
        predictions = model.predict(test)
        pred = np.argmax(predictions, axis=1)
        respective_accuracy = [0, 0, 0, 0, 0]
        total = [0, 0, 0, 0, 0]
        label_class = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
        accuracy = 0
        unknow = 0
        unknow_T = 0
        dnn_unknown = []
        # threshold = threshold * 0.9
        for i in range(len(predictions)):
            if np.max(predictions[i]) < threshold:
                dnn_unknown.append(i)
            if test_label[i] == 0:
                total[0] += 1
                if np.argmax(predictions[i]) == 0:
                    respective_accuracy[0] += 1
                    accuracy += 1
            if test_label[i] >= 1 and test_label[i] < 6:
                total[1] += 1
                if np.argmax(predictions[i]) >= 1 and np.argmax(predictions[i]) < 6:
                    respective_accuracy[1] += 1
                    accuracy += 1
            if test_label[i] >= 6 and test_label[i] < 10:
                total[2] += 1
                if np.argmax(predictions[i]) >= 6 and np.argmax(predictions[i]) < 10:
                    respective_accuracy[2] += 1
                    accuracy += 1
            if test_label[i] >= 10 and test_label[i] < 12:  # R2L
                total[3] += 1
                if np.argmax(predictions[i]) >= 10 and np.argmax(predictions[i]) < 12:
                    respective_accuracy[3] += 1
                    accuracy += 1
            if test_label[i] == 12:
                total[4] += 1
                if np.argmax(predictions[i]) == 12:
                    respective_accuracy[4] += 1
                    accuracy += 1
        total_accuracy = accuracy / len(predictions)
        print('total accuracy', total_accuracy)
        xtimeAcc += total_accuracy

    print(x, 'time total accuracy: ', xtimeAcc / x)
    for i in range(5):
        print('\n{:<6}  :{}'.format(
            label_class[i], respective_accuracy[i] / total[i]))

    dnn_unknown = pd.Series(dnn_unknown)
    dnn_unknown.to_csv(f'{file_path}dnn_unknown_num.csv')

    # 20 time total accuracy:  0.8403029765337354
    pred_label = []
    pred_label1 = []

    for i in range(len(pred)):

        if pred[i] == 0:
            pred_label.append(0)
            pred_label1.append(0)

        if pred[i] >= 1 and pred[i] < 6:
            pred_label.append(1)
            pred_label1.append(1)

        if pred[i] >= 6 and pred[i] < 10:
            pred_label.append(2)
            pred_label1.append(1)

        if pred[i] >= 10 and pred[i] < 12:
            pred_label.append(3)
            pred_label1.append(1)

        if pred[i] == 12:
            pred_label.append(4)
            pred_label1.append(1)
    # 印混淆矩陣
    confussion_matrix = confusion_matrix(
        cm_label, pred_label, labels=[0, 1, 2, 3, 4])
    plot_confusion_matrix(cm=confussion_matrix, normalize=False, target_names=["Normal", "DoS", "Probe", "R2L", "U2R"],
                          title="Confusion Matrix")
    confussion_matrix = confusion_matrix(cm_label1, pred_label1, labels=[0, 1])
    plot_confusion_matrix(cm=confussion_matrix, normalize=False, target_names=["Normal", "Attack"],
                          title="Confusion Matrix")


def Combine():
    data1 = pd.read_csv(f'{file_path}ae_unknown_num.csv')
    data2 = pd.read_csv(f'{file_path}dnn_unknown_num.csv')

    a = data1.iloc[:, 1]
    b = data2.iloc[:, 1]

    a = pd.Series(a)
    b = pd.Series(b)

    c = pd.concat([a, b], axis=0)
    c = pd.DataFrame(c)
    c.columns = ['col1']
    c = c.sort_values(by=['col1'])
    c = c.drop_duplicates()
    c.reset_index(inplace=True, drop=False)
    c.drop(c.columns[[0]], axis=1, inplace=True)
    c.to_csv(f'{file_path}unknown_num.csv')
    # 合併後為幾筆資料
    print('目前未知數量：', len(c))
    return(len(c))


def DBSCANN():

    test = pd.read_csv(f'{file_path}preKDDtest.csv')
    test.drop(test.columns[[-1]], axis=1, inplace=True)
    test.drop(test.columns[[0]], axis=1, inplace=True)
    data = pd.read_csv(f'{file_path}unknown_num.csv')

    k = 0
    a = len(data)
    arr = [0] * a

    for i in range(a):
        arr[k] = test.loc[i]
        k = k + 1
    arr = np.array(arr)

    # 標準化
    # scale_arr = scale(arr)
    scale_arr = MinMaxScaler(feature_range=(0, 1)).fit_transform(arr)

    clustering = DBSCAN(eps=0.5, min_samples=8)
    clustering.fit_predict(scale_arr)

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

    newarr.to_csv(f'{file_path}dbscan.csv')


def GroupDecision():

    eps = 0.5
    data = pd.read_csv(f'{file_path}dbscan.csv')
    label = data.iloc[:, -1]
    data.drop(data.columns[[-1]], axis=1, inplace=True)
    newData = scale_data = MinMaxScaler(
        feature_range=(0, 1)).fit_transform(data)
    pca = PCA(n_components=8)
    newData = pca.fit_transform(scale_data)

    def w(p):
        if (p < eps):
            return 1
        else:
            return 0

    def densityP(p, ne, j):
        total = 0
        for i in ne:
            if (i != j):
                total += w(p[i]) / p[i]
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

    test = pd.read_csv(f'{file_path}preKDDtest.csv')
    test.drop(test.columns[[-1]], axis=1, inplace=True)
    test.drop(test.columns[[0]], axis=1, inplace=True)
    data = pd.read_csv(f'{file_path}unknown_num.csv')

    k = 0
    a = len(data)
    arr = [0] * a

    for i in range(a):
        arr[k] = test.loc[i]
        k = k + 1
    arr = np.array(arr)

    newarr = pd.DataFrame(arr)
    newarr['label'] = newlabel
    newarr.to_csv(f'{file_path}group_decision.csv')


# 混淆矩陣
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}'.format(accuracy))
    plt.show()


# 執行
main()
