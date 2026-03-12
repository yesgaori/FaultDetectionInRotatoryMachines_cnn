import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QPixmap
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

from_window = uic.loadUiType('./archive.ui')[0] # ui를 class로 바꾸는것

# QWidget 기본적인 위젯으로써의 기능
# Exam 은 QWidget class 와 from_window class 를 상속함
class Exam(QWidget, from_window):
    def __init__(self):
        super().__init__()
        # 조상 class init
        self.path = None
        self.setupUi(self)
        self.model = load_model('./spec_rotor_0.9789915680885315.h5')
        self.pushButton.clicked.connect(self.button_slot)
        # button을 실행하면 button_slot과 연결됨
        # signal slot system
        # 경로추적은 OS의 파일매니저를 쓰면 된다!
        # 운영체제가 제공해주는 기본적인 기능들을 응용프로그램 이라고 한다.
    def button_slot(self):
        self.path = QFileDialog.getOpenFileName(self, 'Open file', '/media/user13/data/archive',
                                    'Image Files(*.png);;All File(*.*)')
        print(self.path)
        pixmap = QPixmap(self.path[0])
        self.label_2.setPixmap(pixmap)

        try:
            img = Image.open(self.path[0])
            img = img.convert('RGB')
            img = img.resize((128,128))
            data = np.asarray(img)
            data = data / 255
            data = data.reshape(1, 128, 128, 3)
            predict_value = self.model.predict(data)
            print(predict_value)
            categories = ['정상(Healthy)', '회전체 고장(10%)', '회전체 고장(30%)', '회전체 고장(60%)']
            # 3. 가장 높은 확률의 인덱스(번호) 찾기
            predicted_index = np.argmax(predict_value)  # 예: 1 (두 번째가 1등이면)
            self.label_3.setText(categories[predicted_index])

        except:
            print('error')

# QApplication 객체 만듬
# app.exec_() 이 실행하면 mainWindow 호출
# sys.exit(app.exec_()) 는 앱이 실행중일 땐 동작안함, 실행이 끝나면 리턴값에 의해 종료됨
app = QApplication(sys.argv)
mainWindow = Exam()
mainWindow.show()
sys.exit(app.exec_())
