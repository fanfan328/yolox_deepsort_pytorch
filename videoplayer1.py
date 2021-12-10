from PyQt5.QtCore import QDir, Qt, QUrl, QTimer
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget, QComboBox)
from PyQt5.QtWidgets import QMainWindow,QWidget, QPushButton, QAction, QProgressBar, QLabel, QMessageBox
from PyQt5.QtGui import QIcon
from random import randint

import sys
import os
from object_tracker import ObjectTracker
# from final_app import ObjectTracker

Stylesheet = '''
#StyleProgressBar {
    text-align: center;
    border-radius: 4px;
}
#StyleProgressBar::chunk {
    background-color: #ACf49C;
}

#StyleBtn {
    border-radius: 4px;
    background-color: #2196F3;
    font-size: 18px;
}

#StyleBtn::hover {
  background-color: white; 
  color: black;
  border: 2px solid #2196F3;
}

#StyleCombo {font-size: 18px;}

#StyleLab{font-size: 12pt;}
'''
class ProgressBar(QProgressBar):
    def __init__(self, *args, **kwargs):
        super(ProgressBar, self).__init__(*args, **kwargs)
        self.setValue(0)

    def onTimeout(self):
        if self.value() >= 100:
            self.timer.stop()
            self.timer.deleteLater()
            del self.timer
            return
        self.setValue(self.value() + 1)
    
    def startCounting(self):
        if self.minimum() != self.maximum():
            self.timer = QTimer(self, timeout=self.onTimeout)
            # self.timer.start(randint(1, 3) * 1000)
    
    def setVal(self, data_in):
        self.setValue(data_in)

class VideoWindow(QMainWindow):

    def __init__(self, parent=None):
        super(VideoWindow, self).__init__(parent)
        self.setWindowTitle("Highlighting Basketball Player")
        
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        videoWidget = QVideoWidget()

        self.openButton = QPushButton("Open Video")
        self.openButton.setObjectName("StyleBtn")
        self.openButton.setToolTip("Choose the Video")
        self.openButton.setStatusTip("Open Video")
        self.openButton.setFixedHeight(40)
        self.openButton.setFixedWidth(120)
        self.openButton.clicked.connect(self.openF)

        self.cropButton = QPushButton("Crop Video on Specified Person")
        self.cropButton.setObjectName("StyleBtn")
        self.cropButton.setFixedHeight(40)
        self.cropButton.setFixedWidth(280)
        self.cropButton.clicked.connect(self.confirmation_win_cropping)

        self.combo = QComboBox()
        self.combo.setObjectName("StyleCombo")
        # self.combo.addItem("person1")
        # self.combo.addItem("person2")
        # self.combo.addItem("person3")
        self.combo.setFixedHeight(40)
        # cropButton.setFixedWidth(280)

        #font = QFont('Arial',15)
        #self.combo.setFont(font)

        self.processButton = QPushButton("Process Video")
        self.processButton.setToolTip("Process Video Video")
        self.processButton.setStatusTip("Process Video")
        self.processButton.setObjectName("StyleBtn")
        self.processButton.setFixedHeight(40)
        self.processButton.setFixedWidth(120)
        self.processButton.clicked.connect(self.confirmation_win)
        self.progressBar = ProgressBar(self, minimum=0, maximum=100, objectName="StyleProgressBar")
        self.progressBar.setFixedHeight(40)

        self.playButton = QPushButton("")
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Maximum)

        self.label = QLabel("Filename")
        self.label.setFixedHeight(40)
        self.label.setObjectName("StyleLab")

        # Create a widget for window contents
        wid = QWidget(self)
        self.setCentralWidget(wid)

        # Create layouts to place inside widget
        self.playerLayout = QHBoxLayout()
        # self.playerLayout.setContentsMargins(2, 3, 1, 1)
        # playerLayout.addWidget(openButton)
        self.playerLayout.addWidget(self.playButton)
        self.playerLayout.addWidget(self.positionSlider)

        self.progressLayout = QHBoxLayout()
        self.progressLayout.setContentsMargins(2, 10, 1, 1)
        self.progressLayout.addWidget(self.processButton)
        self.progressLayout.addWidget(self.progressBar)
        
        self.selectorLayout = QHBoxLayout()
        self.selectorLayout.setContentsMargins(2, 10, 1, 1)
        self.selectorLayout.addWidget(self.combo)
        self.selectorLayout.addWidget(self.cropButton)

        self.openLayout = QHBoxLayout()
        self.openLayout.addWidget(self.openButton)
        self.openLayout.addWidget(self.label)
        
        layout = QVBoxLayout()
        layout.addWidget(videoWidget)
        layout.addLayout(self.playerLayout)
        # layout.addWidget(self.errorLabel)
        layout.addLayout(self.openLayout)
        layout.addLayout(self.progressLayout)
        layout.addLayout(self.selectorLayout)

        # Set widget to contain window contents
        wid.setLayout(layout)

        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.currentMediaChanged.connect(self.info_success_win)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)

    def crop(self):
        text = str(self.combo.currentText())
        id= text.split(" ")[-1]
        if id.isnumeric():
            print(f"Target ID = {int(id)}")
            cropped_video_path = self.tracker.crop_vid(int(id))
            if(os.path.isfile(cropped_video_path)):
                self.mediaPlayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(cropped_video_path)))
                self.playButton.setEnabled(True)
            else:
                self.msg_box_win('Err',"Cropping Video Error, Please try to process the video.")
                print("Cropping Video Error, Please try to process the video")
        else:
            self.msg_box_win('Err',"ID Is not found. Please Repeat the Video Processing.")
        #execfile('videoplayer.py')
        # self.close()

    def confirmation_win_cropping(self): 
        msg = QMessageBox()
        msg.setWindowTitle("Confirmation")
        msg.setText("Are you sure to continue the process ?")
        msg.setIcon(QMessageBox.Question)
        msg.setStandardButtons(QMessageBox.Ok|QMessageBox.Cancel)
        msg.button(msg.Ok).clicked.connect(self.crop)

        x = msg.exec_()
        

    def openF(self):
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.homePath(), 'Video Files (*.mp4 *.avi)')

        if self.fileName != '':
            self.label.setText(self.fileName)
            self.mediaPlayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(self.fileName)))
            self.playButton.setEnabled(True)

    def exitCall(self):
        self.close()

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.errorLabel.setText("Error: " + self.mediaPlayer.errorString())

    def info_success_win(self): #Message box after success load the video
        msg = QMessageBox()
        msg.setWindowTitle("Info")
        msg.setText("Load Video Sucess !")
        msg.setIcon(QMessageBox.Information)
        x = msg.exec_()
    
    def msg_box_win(self, type, text): #Message box after success load the video
        msg = QMessageBox()
        msg.setWindowTitle("Info")
        msg.setText(text)
        if(type=='Warn'):
            msg.setIcon(QMessageBox.Warning)
        elif(type=='Err'):
            msg.setIcon(QMessageBox.Critical)
        else:
            msg.setIcon(QMessageBox.Information)
        x = msg.exec_()

    def confirmation_win(self): 
        msg = QMessageBox()
        msg.setWindowTitle("Confirmation")
        msg.setText("Are you sure to continue the process ?")
        msg.setIcon(QMessageBox.Question)
        msg.setStandardButtons(QMessageBox.Ok|QMessageBox.Cancel)
        msg.button(msg.Ok).clicked.connect(self.sendtoNet)

        x = msg.exec_()

    def sendtoNet(self):
        #Initiate the progress bar
        self.progressBar.setVal(0)

        #Starting the network
        # tracker.tes()
        if(self.fileName):
            self.tracker = ObjectTracker()
            tmp = self.fileName.split("/")
            list_id_person, out_video = self.tracker.track_video(tmp[-1], 2, 751, self.progressBar)
            if(self.progressBar.value()==100 or self.progressBar.value()=="100" or len(list_id_person)>1):
                self.progressBar.setVal(100)
                # print("Adding item into Combobox")
                # print(list_id_person)
                # print(out_video)
                if(os.path.isfile(out_video)):
                    self.mediaPlayer.setMedia(
                        QMediaContent(QUrl.fromLocalFile(out_video)))
                    self.playButton.setEnabled(True)
                    for id_person in list_id_person:
                        self.combo.addItem("Person "+str(id_person))
                else:
                    self.msg_box_win('Err',"Process Video Doesn't Exist, Please try to process the video.")
                    print("Process Video Doesn't Exist, Please try to process the video.")
            else:
                print(self.progressBar.value())
                self.msg_box_win('Err',"Failed to Process data")
                print("Failed to Process, Force Close")
                sys.exit(self.close())
            # self.progressBar.setVal(self.tracker.progress)
            # if
            
        
        print(f"Send to the Network")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(Stylesheet)
    player = VideoWindow()
    player.resize(1000, 800)
    player.show()
    sys.exit(app.exec_())

    
