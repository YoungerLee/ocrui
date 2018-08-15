import tensorflow as tf
import numpy as np
import os, sys
import cv2
sys.path.append(os.getcwd())
from ..lib.networks.factory import get_network
from ..lib.fast_rcnn.test import test_ctpn
from ..lib.text_connector.detectors import TextDetector

CANNY_LOW_THRESHOLD = 30  # canny算法低阈值
CANNY_HIGH_THRESHOLD = 60  # canny算法高阈值
HOUGH_DELTARHO = 1  # hough检测步长
HOUGH_DELTA_THETA = 100
class Detector(object):
    def __init__(self, checkpoints='/home/deeple/project/ocrui/detect/checkpoints/stickmobilev1'):
        # create graph
        self.__graph = tf.Graph()
        # init session
        config = tf.ConfigProto(allow_soft_placement=True)
        self.__sess = tf.Session(graph=self.__graph, config=config)
        with self.__sess.as_default():
            with self.__graph.as_default():
                # load network
                with tf.device('/cpu:0'):
                    self.__net = get_network("Mobilenet_test")
                self.__sess.run(tf.global_variables_initializer())
                print(('Loading network {:s}... '.format("Mobilenet_test")), end=' ')
                _, ckpt_file = self.__load_checkpoints(checkpoints)
                saver = tf.train.Saver(tf.global_variables())
                try:
                    print('Restoring from {}...'.format(ckpt_file), end=' ')
                    saver.restore(self.__sess, ckpt_file)
                    print('done')
                except:
                    raise 'Check your pretrained {:s}'.format(ckpt_file)

    def __load_checkpoints(self, checkpoints):
        file_list = os.listdir(checkpoints)
        meta_files = [item for item in file_list if item.endswith('.meta')]
        if len(meta_files) == 0:
            raise ValueError('No meta file found in model directory (%s)' % checkpoints)
        elif len(meta_files) > 1:
            raise  ValueError('There should not be more than one meta file in the model directory (%s)' % checkpoints)
        meta_file = os.path.join(checkpoints, meta_files[0])
        ckpt_file = meta_file.replace('.meta', '')
        return meta_file, ckpt_file


    # 旋转矫正
    def __rotate_calib(self, srcimage):
        blurimage = cv2.blur(srcimage, (3, 3))
        grayimage = cv2.cvtColor(blurimage, cv2.COLOR_BGR2GRAY)
        _, binimage = cv2.threshold(grayimage, 15, 255, cv2.THRESH_BINARY)
        cannyimage = cv2.Canny(binimage, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD, apertureSize=3)
        lines = cv2.HoughLinesP(cannyimage, 1, np.pi / 180, 160, minLineLength=200, maxLineGap=180)
        #        寻找长度最长的线
        if not lines is None:
            distance = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dis = np.sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))
                distance.append(dis)
            max_dis_index = distance.index(max(distance))
            max_line = lines[max_dis_index]
            x1, y1, x2, y2 = max_line[0]

            # 获取旋转角度
            angle = cv2.fastAtan2((y2 - y1), (x2 - x1))
            centerpoint = (srcimage.shape[1] / 2, srcimage.shape[0] / 2)
            rotate_mat = cv2.getRotationMatrix2D(centerpoint, angle, 1.0)  # 获取旋转矩阵
            inverse_rotate_mat = cv2.getRotationMatrix2D(centerpoint, -angle, 1.0)  # 获取反转矩阵
            correct_image = cv2.warpAffine(srcimage, rotate_mat, (srcimage.shape[1], srcimage.shape[0]),
                                           borderValue=(0, 0, 0))
            return correct_image, inverse_rotate_mat
        else:
            return srcimage, None
    def detect(self, img):
        rotated_img, rotate_mat = self.__rotate_calib(img)
        scores, boxes = test_ctpn(self.__sess, self.__net, rotated_img)
        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        labeled_img, coord_rotated = self.__draw_boxes(img, boxes, rotate_mat)
        return labeled_img, rotated_img, coord_rotated

    def __draw_boxes(self, img, boxes, rotate_mat):
        coord = []
        img_labeled = np.zeros(img.shape)
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            rotated_box = np.zeros(8)
            # 旋转后图像的四点坐标
            if not rotate_mat is None:
                [[rotated_box[0]], [rotated_box[1]]] = np.dot(rotate_mat, np.array([[box[0]], [box[1]], [1]]))
                [[rotated_box[2]], [rotated_box[3]]] = np.dot(rotate_mat, np.array([[box[2]], [box[3]], [1]]))
                [[rotated_box[4]], [rotated_box[5]]] = np.dot(rotate_mat, np.array([[box[4]], [box[5]], [1]]))
                [[rotated_box[6]], [rotated_box[7]]] = np.dot(rotate_mat, np.array([[box[6]], [box[7]], [1]]))

                img_labeled = cv2.line(img, (int(rotated_box[0]), int(rotated_box[1])), (int(rotated_box[2]), int(rotated_box[3])), color, 2)
                img_labeled = cv2.line(img_labeled, (int(rotated_box[0]), int(rotated_box[1])), (int(rotated_box[4]), int(rotated_box[5])), color, 2)
                img_labeled = cv2.line(img_labeled, (int(rotated_box[6]), int(rotated_box[7])), (int(rotated_box[2]), int(rotated_box[3])), color, 2)
                img_labeled = cv2.line(img_labeled, (int(rotated_box[4]), int(rotated_box[5])), (int(rotated_box[6]), int(rotated_box[7])), color, 2)
            else:
                img_labeled = cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                img_labeled = cv2.line(img_labeled, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
                img_labeled = cv2.line(img_labeled, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
                img_labeled = cv2.line(img_labeled, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)
            min_x = min(int(box[0]), int(box[2]), int(box[4]), int(box[6]))
            min_y = min(int(box[1]), int(box[3]), int(box[5]), int(box[7]))
            max_x = max(int(box[0]), int(box[2]), int(box[4]), int(box[6]))
            max_y = max(int(box[1]), int(box[3]), int(box[5]), int(box[7]))
            coord.append([min_x, min_y, max_x, max_y])
        return img_labeled, coord

    def closeSess(self):
        self.__sess.close()

if __name__ == '__main__':
    Detector()
