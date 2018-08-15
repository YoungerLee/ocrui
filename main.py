import os
import glob
import sys
import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from ui_main import Ui_MainWindow
from text_end_to_end import TextEnd2End

sys.path.append(os.getcwd())
this_dir = os.path.dirname(__file__)

from detect.lib.utils.timer import Timer
from train_thread import TrainDetectThread, TrainRecognizeThread

class MainQt(QMainWindow, Ui_MainWindow):
    '''
    继承ui类并进行界面逻辑操作
    '''
    train_detect_signal = pyqtSignal()  # 启动或停止训练的信号
    train_recognize_signal = pyqtSignal()
    def __init__(self, parent=None):
        super(MainQt, self).__init__(parent)
        self.__img_paths = []
        self.__curr = 0
        self.__img_num = 0
        self.__text_cnn = TextEnd2End()
        #--训练检测线程--
        self.train_detect_thread = TrainDetectThread()
        # 连接训练停止操作
        self.train_detect_signal.connect(self.train_detect_thread.stop_thread)
        # 打印日志信号
        self.train_detect_thread.update_log.connect(self.handleDisplay)
        # 绘图信号
        self.train_detect_thread.update_fig.connect(self.handlePlot)
        #--训练识别线程--
        self.train_recognize_thread = TrainRecognizeThread()
        # 连接训练停止操作
        self.train_recognize_signal.connect(self.train_recognize_thread.stop_thread)
        # 打印日志信号
        self.train_recognize_thread.update_log.connect(self.handleDisplay_2)
        # 绘图信号
        self.train_recognize_thread.update_fig.connect(self.handlePlot_2)
        # 初始化界面
        self.setupUi(self)
        self.initUI()

    def initUI(self):
        '''
        初始化信号与槽的连接
        :return: None
        '''
        self.chooseFolder.clicked.connect(self.onResponseBtn)
        self.nextButton.clicked.connect(self.onResponseBtn)
        self.lastButton.clicked.connect(self.onResponseBtn)
        self.saveResult.clicked.connect(self.onResponseBtn)

        self.chooseTrainDirButton.clicked.connect(self.onResponseBtn)
        self.chooseOutputFolder.clicked.connect(self.onResponseBtn)
        self.beginButton.clicked.connect(self.onResponseBtn)
        self.saveButton.clicked.connect(self.onResponseBtn)

        self.chooseTrainFileButton.clicked.connect(self.onResponseBtn)
        self.chooseTestFileButton.clicked.connect(self.onResponseBtn)
        self.chooseOutputFolder_2.clicked.connect(self.onResponseBtn)
        self.beginButton_2.clicked.connect(self.onResponseBtn)
        self.saveButton_2.clicked.connect(self.onResponseBtn)

        self.detectLossView.mpl.fig.suptitle('total loss/iter')
        self.recognizetLossView.mpl.fig.suptitle('ctc_loss/iter')

    # 定义槽函数
    def loadImage(self):
        if not ((self.filePath.text() is None) or (self.filePath.text() == '')):
            timer = Timer()
            img_path = self.__img_paths[self.__curr]
            img_name = img_path.split('/')[-1]
            timer.tic()
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
            img_labeled, bbox_str, pred_str = self.__text_cnn.get_result(img)
            show = cv2.resize(img_labeled, (1280, 960), interpolation=cv2.INTER_CUBIC)
            timer.toc()
            show_img = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.image_show.setPixmap(QPixmap.fromImage(show_img))
            self.fileNameText.setText(img_name)
            self.speedText.setText('{:.4f}s'.format(timer.total_time))
            self.locationText.setText(bbox_str)
            self.resultText.setText(pred_str)

    def initDetectTrain(self):
        # 获取训练参数
        train_data_dir = str(self.trainFileEdit.text())
        batch_size = int(self.bnSpinBox.value())
        solver = str(self.optComboBox.currentText())
        learning_rate = float(self.lrComboBox.currentText())
        snapshot = float(self.snapSpinBox.value())
        output_dir = str(self.outFileEdit.text())
        pretrained = bool(self.pretrainCheckBox.isChecked())
        # 设置训练参数
        self.train_detect_thread.setOutputDir(output_dir)
        self.train_detect_thread.setBatchSize(batch_size)
        self.train_detect_thread.setSolver(solver)
        self.train_detect_thread.setLearningRate(learning_rate)
        self.train_detect_thread.setSnapshot(snapshot)
        self.train_detect_thread.setPretrained(pretrained)
        # 启动线程
        self.train_detect_thread.start()

    def initRecognizeTrain(self):
        # 获取训练参数
        train_file = str(self.trainFileEdit_2.text())
        test_file = str(self.testFileEdit.text())
        pretrained = bool(self.pretrainCheckBox_2.isChecked())
        batch_size = int(self.bnSpinBox_2.value())
        solver = str(self.optComboBox_2.currentText())
        learning_rate = float(self.lrComboBox_2.currentText())
        snapshot = float(self.snapSpinBox_2.value())
        output_dir = str(self.outFileEdit_2.text())
        # 设置训练参数
        self.train_recognize_thread.setOutputDir(output_dir)
        self.train_recognize_thread.setBatchSize(batch_size)
        self.train_recognize_thread.setSolver(solver)
        self.train_recognize_thread.setLearningRate(learning_rate)
        self.train_recognize_thread.setSnapshot(snapshot)
        self.train_recognize_thread.setPretrained(pretrained)
        self.train_recognize_thread.setTrainFile(train_file)
        self.train_recognize_thread.setTestFile(test_file)
        # 启动线程
        self.train_recognize_thread.start()

    def onResponseBtn(self):
        btn = self.sender()
        btn_name = btn.objectName()
        if btn_name == 'chooseFolder':
            dir_path = QFileDialog.getExistingDirectory(self, 'choose folder', './')
            self.filePath.setText(dir_path)
            self.__img_paths.clear()
            img_paths = glob.glob(os.path.join(dir_path, '*.png')) + \
                        glob.glob(os.path.join(dir_path, '*.jpg'))
            self.__img_paths.extend(img_paths)
            self.__img_paths.sort()
            self.__curr = 0
            self.__img_num = len(img_paths)
            self.loadImage()
        elif btn_name == 'nextButton':
            if self.__curr < self.__img_num - 1:
                self.__curr += 1
                self.loadImage()
        elif btn_name == 'lastButton':
            if self.__curr > 0:
                self.__curr -= 1
                self.loadImage()
        elif btn_name == 'saveResult':
            img_dir = str(self.filePath.text())
            if not (img_dir is None or img_dir == ''):
                img_name = str(self.fileNameText.text())
                img_path = os.path.join(img_dir, img_name)
                location = str(self.locationText.text())
                result = str(self.resultText.text())
                filename, ok = QFileDialog.getSaveFileName(self, 'save file', './untitled.txt',
                                                           'All Files (*);;Text Files (*.txt)')
                if ok:
                    with open(filename, 'w') as f:
                        f.write('img_path: ')
                        f.write(img_path)
                        f.write('\n')
                        f.write('location: ')
                        f.write(location)
                        f.write('\n')
                        f.write('text: ')
                        f.write(result)
                        f.write('\n')
        elif btn_name == 'chooseTrainDirButton':
            dir_path = QFileDialog.getExistingDirectory(self, 'choose folder', './')
            self.trainFileEdit.setText(dir_path)
        elif btn_name == 'chooseOutputFolder':
            dir_path = QFileDialog.getExistingDirectory(self, 'choose folder', './')
            self.outFileEdit.setText(dir_path)
        elif btn_name == 'beginButton':
            if self.beginButton.text() == 'begin':
                self.beginButton.setText('stop')
                # 清除历史数据
                self.log_output.clear()
                self.detectLossView.dataY.clear()
                self.detectLossView.dataX.clear()
                self.initDetectTrain()
            elif self.beginButton.text() == 'stop':
                self.train_detect_signal.emit()
                self.beginButton.setText('begin')
                self.train_detect_thread.quit()
        elif btn_name == 'saveButton':
            save_path = QFileDialog.getExistingDirectory(self, 'choose save folder', './')
            logs = str(self.log_output.toPlainText())
            logs_path = os.path.join(save_path, 'train_logs.txt')
            loss_img_path = os.path.join(save_path, 'loss_iter.jpg')
            with open(logs_path, 'w') as f:
                f.write(logs)
                f.write('\n')
            self.detectLossView.mpl.fig.savefig(loss_img_path)
        elif btn_name == 'chooseTrainFileButton':
            fileName, _ = QFileDialog.getOpenFileName(self,
                                                      'choose file',
                                                      './',
                                                      'All Files (*)')  # 设置文件扩展名过滤,注意用双分号间隔
            self.trainFileEdit_2.setText(fileName)
        elif btn_name == 'chooseTestFileButton':
            fileName, _ = QFileDialog.getOpenFileName(self,
                                                      'choose file',
                                                      "./",
                                                      "All Files (*)")  # 设置文件扩展名过滤,注意用双分号间隔
            self.testFileEdit.setText(fileName)
        elif btn_name == 'chooseOutputFolder_2':
            dir_path = QFileDialog.getExistingDirectory(self, 'choose folder', './')
            self.outFileEdit_2.setText(dir_path)
        elif btn_name == 'beginButton_2':
            if self.beginButton_2.text() == 'begin':
                self.beginButton_2.setText('stop')
                # 清除历史数据
                self.log_output_2.clear()
                self.recognizetLossView.dataY.clear()
                self.recognizetLossView.dataX.clear()
                self.initRecognizeTrain()
            elif self.beginButton_2.text() == 'stop':
                self.train_recognize_signal.emit()
                self.beginButton_2.setText('begin')
                self.train_recognize_thread.quit()
        elif btn_name == 'saveButton_2':
            save_path = QFileDialog.getExistingDirectory(self, 'choose save folder', './')
            logs = str(self.log_output_2.toPlainText())
            logs_path = os.path.join(save_path, 'train_logs.txt')
            loss_img_path = os.path.join(save_path, 'loss_iter.jpg')
            with open(logs_path, 'w') as f:
                f.write(logs)
                f.write('\n')
            self.recognizetLossView.mpl.fig.savefig(loss_img_path)

    def handleDisplay(self, data):
        self.log_output.append(data)

    def handlePlot(self, datay, datax):
        self.detectLossView.dataX.append(datax)
        self.detectLossView.dataY.append(datay)
        self.detectLossView.mpl.update_figure(self.detectLossView.dataY, self.detectLossView.dataX)

    def handleDisplay_2(self, data):
        self.log_output_2.append(data)

    def handlePlot_2(self, datay, datax):
        self.recognizetLossView.dataX.append(datax)
        self.recognizetLossView.dataY.append(datay)
        self.recognizetLossView.mpl.update_figure(self.recognizetLossView.dataY, self.recognizetLossView.dataX)

    def closeEvent(self, event):
        """
        重写closeEvent方法，实现dialog窗体关闭时执行一些代码
        :param event: close()触发的事件
        :return: None
        """
        reply = QMessageBox.question(self, 'ocrui', 'Are you sure exit?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.__text_cnn.closeProgram()
            self.train_recognize_thread.clossProgram()
            self.train_detect_thread.clossProgram()
            self.train_detect_thread.quit()
            self.train_recognize_thread.quit()
            event.accept()
        else:
            event.ignore()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    win = MainQt()
    win.show()
    sys.exit(app.exec_())
