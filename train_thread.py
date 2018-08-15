from PyQt5.QtCore import QThread, pyqtSignal, QDateTime
import sys
import os

sys.path.append(os.getcwd())
root = os.path.abspath(os.path.join(os.path.dirname(__file__)))

import detect.lib.fast_rcnn.train as detect_solver
import recognize.crnn.solver as recognize_solver
class TrainDetectThread(QThread):
    update_log = pyqtSignal(str)
    update_fig = pyqtSignal(float, int)

    def __init__(self, parent=None):

        super(TrainDetectThread, self).__init__(parent)
        self.__output_dir = os.path.join(root, 'detect/output')
        self.__solver = 'Adam'
        self.__learning_rate = 0.0001
        self.__batch_size = 300
        self.__snapshot = 100
        self.__pretrained = False
        self.__sw = None

    def plot_out(self, datay, datax):
        self.update_fig.emit(datay, datax)

    def print_out(self, data):
        self.update_log.emit(data)
    # 运行训练线程
    def run(self):
        self.__sw = detect_solver.SolverWrapper()
        self.print_out('Loaded dataset `{:s}` for training'.format(self.__sw.imdb.name))
        self.print_out('Output will be saved to `{:s}`'.format(self.__output_dir))
        self.print_out('Logs will be saved to `{:s}`'.format(self.__sw.log_dir))
        self.__sw.train_model(output_dir=self.__output_dir,
                            solver=self.__solver,
                            learning_rate=self.__learning_rate,
                            batch_size=self.__batch_size,
                            snap_iter=self.__snapshot,
                            restore=self.__pretrained,
                            max_iters=50000,
                            print_out=self.print_out,
                            plot_out=self.plot_out)

    def stop_thread(self):
        self.__sw.stop_iter()

    def setOutputDir(self, output_dir):
        self.__output_dir = output_dir

    def setSolver(self, solver):
        self.__solver = solver

    def setLearningRate(self, learning_rate):
        self.__learning_rate = learning_rate

    def setBatchSize(self, batch_size):
        self.__batch_size = batch_size

    def setSnapshot(self, snapshot):
        self.__snapshot = snapshot

    def setPretrained(self, pretrained):
        self.__pretrained = pretrained

    def clossProgram(self):
        if not self.__sw is None:
            self.__sw.closeSess()

class TrainRecognizeThread(QThread):
    update_log = pyqtSignal(str)
    update_fig = pyqtSignal(float, int)

    def __init__(self, parent=None):
        super(TrainRecognizeThread, self).__init__(parent)
        self.__train_file = os.path.join(root, 'recognize/data/train.txt')
        self.__test_file = os.path.join(root, 'recognize/data/test.txt')
        self.__output_dir = os.path.join(root, 'recognize/output')
        self.__solver = 'Adam'
        self.__learning_rate = 0.01
        self.__batch_size = 100
        self.__snapshot = 100
        self.__pretrained = False
        self.__sw = None

    def plot_out(self, datay, datax):
        self.update_fig.emit(datay, datax)

    def print_out(self, data):
        self.update_log.emit(data)

    # 运行训练线程
    def run(self):
        self.__sw = recognize_solver.SolverWrapper()
        self.print_out('Output will be saved to `{:s}`'.format(self.__output_dir))
        self.print_out('Logs will be saved to `{:s}`'.format(self.__sw.log_dir))
        self.__sw.train_model(train_file=self.__train_file,
                              test_file=self.__test_file,
                              output_dir=self.__output_dir,
                              solver=self.__solver,
                              learning_rate=self.__learning_rate,
                              batch_size=self.__batch_size,
                              snap_iter=self.__snapshot,
                              restore=self.__pretrained,
                              max_iters=100000,
                              print_out=self.print_out,
                              plot_out=self.plot_out)

    def stop_thread(self):
        self.__sw.stop_iter()

    def setTrainFile(self, train_file):
        self.__train_file = train_file

    def setTestFile(self, test_file):
        self.__test_file = test_file

    def setOutputDir(self, output_dir):
        self.__output_dir = output_dir

    def setSolver(self, solver):
        self.__solver = solver

    def setLearningRate(self, learning_rate):
        self.__learning_rate = learning_rate

    def setBatchSize(self, batch_size):
        self.__batch_size = batch_size

    def setSnapshot(self, snapshot):
        self.__snapshot = snapshot

    def setPretrained(self, pretrained):
        self.__pretrained = pretrained

    def clossProgram(self):
        if not self.__sw is None:
            self.__sw.closeSess()