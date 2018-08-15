import tensorflow as tf
import numpy as np
import os
from .config import cfg_from_file, cfg
from ..networks.data_layer import TextDataSet
from ..networks.crnn import CRNN
from ..utils.text_encode import LabelConverter
from PIL import Image
from ..utils.timer import Timer

root = '/home/deeple/project/crnn'
text_file = '/home/deeple/project/crnn/data/train.txt'
image_list = []
label_list = []
with open(text_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        image_path, label = line.split(' ')
        image_list.append(image_path)
        label_list.append(label)
samples_num = len(image_list)
cfg_from_file('text.yml')
nclass = len(cfg.ALPHABET) + 1
imgH = 32
imgW = 512
nh = 256
channels = 1
converter = LabelConverter(alphabet=cfg.ALPHABET)

# load images
def load_image(img_path):
    img = Image.open(img_path).convert('L')
    img = img.resize((imgW, imgH))
    img = np.asarray(img)
    img = img / 127.5 - 1.0
    img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)
    return img


with tf.device('/gpu:0'):
    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = CRNN(imgH, nclass, nh)
    # load model
    print(('Loading network {:s}... '.format("Mobilenet_test")), end=' ')
    saver = tf.train.Saver()
    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    pred = net.get_output('lstm_o')
    acc_num = 0
    for i in range(samples_num):
        img_path = image_list[i]
        label_str = label_list[i]
        timer = Timer()
        timer.tic()
        img = load_image(img_path)
        feed_dict = {
            net.data: img
        }
        pred_val = sess.run(pred, feed_dict)
        preds = np.argmax(pred_val, axis=2)
        preds = preds.squeeze(1)
        pred_str_raw = converter.decode_sequence([list(preds)], [len(preds)], raw=True)
        pred_str_true = converter.decode_sequence([list(preds)], [len(preds)], raw=False)
        timer.toc()
        print('running time: %fs' % timer.total_time)
        print('predict: %-20s  actual: %-20s' % (pred_str_raw, pred_str_true))
        if label_str == pred_str_true:
            acc_num += 1
    print('-------------------------------------------')
    print('accuracy: %f' % (acc_num * 100.0 / samples_num))
    print('-------------------------------------------')
