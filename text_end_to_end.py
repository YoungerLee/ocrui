from detect.ctpn.detector import Detector
from recognize.crnn.recognizer import Recognizer
import cv2

class TextEnd2End(object):
    '''
    端对端光学字符识别
    '''
    def __init__(self):
        self.__detector = Detector()    # 检测器
        self.__recognizer = Recognizer()    # 识别器

    def get_result(self, img):
        '''

        :param img: 待识别图像
        :return: labeled_img 标注字符区域的图像
                 bbox_str 字符区域坐标值
                 pred_str 识别的字符串
        '''
        # 字符区域检测
        labeled_img, rotated_img, bboxes = self.__detector.detect(img)
        bbox = bboxes[0]
        roi = rotated_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGBA2GRAY)
        roi_norm = cv2.resize(roi_gray, (512, 32), interpolation=cv2.INTER_CUBIC)
        # 字符识别
        pred_str = self.__recognizer.recogize(roi_norm)
        bbox_str = ','.join([str(i) for i in bbox])
        return labeled_img, bbox_str, pred_str

    def closeProgram(self):
        self.__detector.closeSess()
        self.__recognizer.closeSess()