import sys
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout

from QSwitchControl import SwitchControl


class Form(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(400, 400)
        self.setWindowTitle("SwitchControl test")
        self.setStyleSheet("""
		background-color: #222222;
		""")
        switch_control = SwitchControl()
        switch_control.setGeometry(QtCore.QRect(0, 0, 61, 61))
        hbox = QHBoxLayout()
        hbox.addWidget(switch_control, Qt.AlignLeft, Qt.AlignTop)
        self.setLayout(hbox)
        self.show()


app = QApplication(sys.argv)
form = Form()
if __name__ == '__main__':
    sys.exit(app.exec_())
