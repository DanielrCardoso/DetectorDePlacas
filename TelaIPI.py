import sys, os 
import Main
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout,QPushButton
from PyQt5.QtCore import Qt 
from PyQt5.QtGui import QPixmap 

class ImageLabel(QLabel): 
    def __init__(self): 
        super().__init__() 

        self.setAlignment(Qt.AlignCenter) 
        self.setText('\n\n Drop Image Here \n\n') 
        self.setStyleSheet(''' QLabel{ border: 4px dashed #aaa } ''') 

    def setPixmap(self, image): 
        super().setPixmap(image) 
    
class AppDemo(QWidget): 
    def __init__(self): 
        super().__init__()

        self.resize(400, 400) 
        self.setAcceptDrops(True) 
        mainLayout = QVBoxLayout() 
        self.photoViewer = ImageLabel() 
        mainLayout.addWidget(self.photoViewer) 
        self.Botao1 = QPushButton()
        self.Botao1.setGeometry(QtCore.QRect(680, 520, 93, 28))
        self.Botao1.setObjectName("EfetuaProcessamento")
        _translate = QtCore.QCoreApplication.translate
        self.Botao1.setText(_translate("", "Processamento Carro"))
        self.Botao1.clicked.connect(self.Clicou1)
        mainLayout.addWidget(self.Botao1)
        self.Botao2 = QPushButton()
        self.Botao2.setGeometry(QtCore.QRect(680, 520, 93, 28))
        self.Botao2.setObjectName("EfetuaProcessamento")
        _translate = QtCore.QCoreApplication.translate
        self.Botao2.setText(_translate("", "Processamento Placa"))
        self.Botao2.clicked.connect(self.Clicou2)
        mainLayout.addWidget(self.Botao2) 
        self.setLayout(mainLayout) 

    def Clicou1(self):
        if file_path:
            Main.Carro(file_path)
        else:
            print("Coloque uma imagem companheiro")
    def Clicou2(self):
        if file_path:
            Main.Placa(file_path)
        else:
            print("Coloque uma imagem companheiro")
    def dragEnterEvent(self, event): 
        if event.mimeData().hasImage: 
            event.accept() 
        else: 
            event.ignore() 

    def dragMoveEvent(self, event): 
        if event.mimeData().hasImage: 
            event.accept()
        else: 
            event.ignore() 

    def dropEvent(self, event): 
        if event.mimeData().hasImage: 
            event.setDropAction(Qt.CopyAction) 
            global file_path  
            file_path = event.mimeData().urls()[0].toLocalFile() 
            print(file_path)
            self.set_image(file_path) 
            event.accept() 
        else: 
            event.ignore() 

    def set_image(self, file_path): 
        self.photoViewer.setPixmap(QPixmap(file_path)) 
             
app = QApplication(sys.argv) 
demo = AppDemo() 
demo.show() 
sys.exit(app.exec())