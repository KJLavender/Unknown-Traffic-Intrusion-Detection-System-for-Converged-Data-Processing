import random
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtMultimedia import QSound
from PyQt5.QtGui import QMovie
import os
import sys
import time
from UI import Ui_MainWindow
from subUI import Ui_SubWindow
from PyQt5.QtCore import QTimer, QRect, QPropertyAnimation
# https://github.com/Prx001/QSwitchControl
from QSwitchControl import SwitchControl
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import scale

current_path = os.path.abspath(__file__)
file_path = os.path.dirname(current_path) + "\\"

AE = keras.models.load_model(f'{file_path}AE_MODEL')
DNN = keras.models.load_model(f'{file_path}DNN_MODEL')
test = pd.read_csv(f'{file_path}preKDDtest.csv')
test.drop(test.columns[[0]], axis=1, inplace=True)
x_test = test.iloc[:, 0:122]
x_test = np.array(x_test)
x_test = scale(x_test)


reconstructions = AE.predict(x_test)
# prediction = DNN.predict(x_test)
test_loss = tf.keras.losses.mae(reconstructions, x_test)
threshold = np.mean(test_loss) + np.std(test_loss)

# 全域變數
HA = 0

red_light = f"{file_path}image/Light-red.png"
green_light = f"{file_path}image/Light-green.png"
png_light = f"{file_path}image/Light-Bulb-PNG-Pic.png"
load_gif = f"{file_path}image/Retrain-unscreen.gif"
train = f"{file_path}image/Train2.png"
buffer = f"{file_path}image/buffer.png"
sound = f"{file_path}image/select07.wav"
datain = f"{file_path}image/datain.png"


# pyuic5 -x UI.ui -o UI.py
# pyuic5 -x subUI.ui -o subUI.py


class MainWindow_controller(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()  # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.buffer.setPixmap(QtGui.QPixmap(buffer))
        self.ui.datain.setPixmap(QtGui.QPixmap(datain))
        self.ui.light.setPixmap(QtGui.QPixmap(png_light))
        self.sound = QSound(sound, self)  # sound
        self.ui.tableWidget.setColumnWidth(0, 300)
        self.ui.switch_control = SwitchControl(
            self.ui.centralwidget, bg_color='#BEBEBE', active_color='#FF79BC', animation_duration=800, checked=False)
        self.ui.switch_control.setGeometry(QtCore.QRect(40, 30, 0, 0))
        self.gate = 0
        self.counters = 2480
        self.ui.counter.setText(str(self.counters))
        self.switch = 0
        self.setup_control()

        self.sub_window = SubWindow()
        self.sub_window.setWindowTitle('Retraining')
        self.sub_window.setFixedSize(606, 312)

    def setup_control(self):
        global HA
        self.qTimerA = QTimer()
        self.qTimerA.setInterval(50)
        self.qTimerA.timeout.connect(self.setup_control)
        self.qTimerA.start()
        if HA == 1:
            self.ui.tableWidget.insertRow(0)
            self.ui.tableWidget.setItem(
                0, 0, QtWidgets.QTableWidgetItem('-----Retrain Done-----'))
            self.ui.tableWidget.item(0, 0).setBackground(
                QtGui.QColor(255, 165, 0))
            HA = 0

        if self.ui.switch_control.isChecked() == True and self.gate == 0:
            self.sound.play()
            self.ui.START.raise_()
            self.gate = 1
            self.StartClicked()
        elif self.ui.switch_control.isChecked() == False and self.gate == 1:
            self.sound.play()
            self.ui.STOP.raise_()
            self.gate = 0
            self.StopClicked()

    def StartClicked(self):
        self.qTimer = QTimer()
        self.qTimer.setInterval(100)  # 1s
        self.qTimer.timeout.connect(self.Time)
        self.qTimer.start()

        self.qTimerData = QTimer()
        self.qTimerData.setInterval(600)  # 0.6s
        self.qTimerData.timeout.connect(self.DataAnimation)
        self.qTimerData.start()

    def StopClicked(self):
        self.ui.tableWidget.insertRow(0)
        self.ui.tableWidget.setItem(
            0, 0, QtWidgets.QTableWidgetItem('STOPPPPPP'))
        self.qTimer.stop()
        self.qTimerData.stop()
        self.ui.light.setPixmap(QtGui.QPixmap(png_light))

    def Time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.ui.tableWidget.insertRow(0)
        self.ui.tableWidget.setItem(
            0, 0, QtWidgets.QTableWidgetItem(localtime))
        X = random.randint(0, len(x_test)-1)
        reconstructions = AE.predict(x_test[[X]])
        test_loss = tf.keras.losses.mae(reconstructions, x_test[[X]])
        # subscreen

        if test_loss > threshold:
            self.ui.tableWidget.setItem(
                0, 1, QtWidgets.QTableWidgetItem('AE未知'))
            self.ui.tableWidget.setItem(
                0, 2, QtWidgets.QTableWidgetItem('----'))
            self.ui.tableWidget.setItem(
                0, 3, QtWidgets.QTableWidgetItem('----'))
            self.ui.light.setPixmap(QtGui.QPixmap(png_light))
            self.counters += 1
            self.ui.counter.setText(str(self.counters))
        else:
            prediction = DNN.predict(x_test[[X]])
            if np.max(prediction) < 0.8:
                self.ui.tableWidget.setItem(
                    0, 1, QtWidgets.QTableWidgetItem('DNN未知'))
                self.ui.tableWidget.setItem(
                    0, 2, QtWidgets.QTableWidgetItem('----'))
                self.ui.tableWidget.setItem(
                    0, 3, QtWidgets.QTableWidgetItem('----'))
                self.ui.light.setPixmap(QtGui.QPixmap(png_light))
                self.counters += 1
                self.ui.counter.setText(str(self.counters))
            else:
                self.ui.tableWidget.setItem(
                    0, 1, QtWidgets.QTableWidgetItem('已知'))
                if np.argmax(prediction) == 0:
                    self.ui.light.setPixmap(QtGui.QPixmap(green_light))
                    self.ui.tableWidget.setItem(
                        0, 2, QtWidgets.QTableWidgetItem('良性'))
                    self.ui.tableWidget.item(0, 2).setBackground(
                        QtGui.QColor(0, 255, 0))
                    self.ui.tableWidget.setItem(
                        0, 3, QtWidgets.QTableWidgetItem('良性'))
                else:
                    self.ui.light.setPixmap(QtGui.QPixmap(red_light))
                    self.ui.tableWidget.setItem(
                        0, 2, QtWidgets.QTableWidgetItem('惡意'))
                    self.ui.tableWidget.item(0, 2).setBackground(
                        QtGui.QColor(255, 0, 0))
                if np.argmax(prediction) >= 1 and np.argmax(prediction) < 6:
                    self.ui.tableWidget.setItem(
                        0, 3, QtWidgets.QTableWidgetItem('DoS'))
                elif np.argmax(prediction) >= 6 and np.argmax(prediction) < 10:
                    self.ui.tableWidget.setItem(
                        0, 3, QtWidgets.QTableWidgetItem('Probe'))
                elif np.argmax(prediction) >= 10 and np.argmax(prediction) < 12:
                    self.ui.tableWidget.setItem(
                        0, 3, QtWidgets.QTableWidgetItem('R2L'))
                elif np.argmax(prediction) == 12:
                    self.ui.tableWidget.setItem(
                        0, 3, QtWidgets.QTableWidgetItem('U2R'))

        if self.counters >= 2500:
            self.change()

    def change(self):
        self.counters = 0
        self.ui.counter.setText(str(self.counters))
        self.sub_window.switch = 1
        self.sub_window.show()
        # self.qTimer.stop()
        # self.qTimerData.stop()
        # self.ui.START.raise_()
        # self.ui.RE.raise_()
        self.ui.tableWidget.insertRow(0)
        self.ui.tableWidget.setItem(
            0, 0, QtWidgets.QTableWidgetItem('-----Retraining-----'))
        self.ui.tableWidget.item(0, 0).setBackground(
            QtGui.QColor(255, 165, 0))

    def DataAnimation(self):
        self.animation = QPropertyAnimation(self.ui.datain, b'geometry')
        self.animation.setDuration(600)  # 0.6s
        self.animation.setStartValue(QRect(380, 30, 51, 61))
        self.animation.setKeyValueAt(0.2, QRect(375, 25, 55, 65))
        self.animation.setKeyValueAt(0.4, QRect(385, 25, 55, 65))
        self.animation.setEndValue(QRect(380, 30, 51, 61))
        self.animation.start()


class SubWindow(QtWidgets.QMainWindow, Ui_SubWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_SubWindow()
        #self.setWindowTitle('Retraining')
        self.ui.setupUi(self)
        self.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint)
        self.ui.background.setPixmap(
            QtGui.QPixmap(train))
        self.ui.movie = QMovie(load_gif)
        self.ui.gif.setMovie(self.ui.movie)
        self.ui.movie.start()
        self.switch = 0
        self.i = 0

        self.qTimer = QTimer()
        self.qTimer.setInterval(25)  #
        self.qTimer.timeout.connect(self.run)
        self.qTimer.start()

    def run(self):
        global HA
        if self.switch == 1:
            self.i += 1
            self.ui.progressBar.setValue(self.i+1)
            self.ui.gif.setGeometry(QtCore.QRect(40+4*self.i, 110, 120, 120))
            if self.i >= 100:
                HA = 1
                self.switch = 0
                self.i = 0
                self.close()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller()
    window.setFixedSize(813, 522)
    window.setWindowTitle('融合式資料處理之未知流量入侵偵測系統')
    window.show()
    sys.exit(app.exec_())
