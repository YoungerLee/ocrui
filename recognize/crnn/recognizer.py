import tensorflow as tf
import numpy as np
from .config import cfg_from_file, cfg
from ..networks.crnn import CRNN
from ..utils.text_encode import LabelConverter
import os

root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
cfg_from_file(os.path.join(root, 'text.yml'))
nclass = len(cfg.ALPHABET) + 1
imgH = 32
imgW = 512
nh = 256
channels = 1

class Recognizer(object):
    def __init__(self, checkpoints='/home/deeple/project/ocrui/recognize/checkpoints/mobilev2'):
        # create graph
        self.__graph = tf.Graph()
        # init session
        self.__sess = tf.Session(graph=self.__graph)
        # init encoder-decoder
        self.__converter = LabelConverter(alphabet=cfg.ALPHABET)
        # load model
        with self.__sess.as_default():
            with self.__graph.as_default():
                # load network
                with tf.device('/cpu:0'):
                    self.__net = CRNN(imgH, nclass, nh)
                self.__sess.run(tf.global_variables_initializer())
                # load saver
                saver = tf.train.Saver(tf.global_variables())
                print(('Loading network {:s}... '.format("Mobilenet_test")), end=' ')
                _, ckpt_file = self.__load_checkpoints(checkpoints)
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

    def recogize(self, img):
        img = np.expand_dims(img, axis=2)
        img = np.expand_dims(img, axis=0)
        pred = self.__net.get_output('lstm2')
        feed_dict = {
            self.__net.data: img
        }
        pred_val = self.__sess.run(pred, feed_dict)
        preds = np.argmax(pred_val, axis=2)
        preds = preds.squeeze(1)
        pred_str = self.__converter.decode_sequence([list(preds)], [len(preds)], raw=False)
        return pred_str

    def closeSess(self):
        self.__sess.close()