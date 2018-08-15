import tensorflow as tf
from .network import Network
from ..fast_rcnn.config import cfg


class Mobilenet_test(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data, 'im_info': self.im_info})
        self.trainable = trainable
        self.setup()

    def setup(self):
        anchor_scales = cfg.ANCHOR_SCALES
        _feat_stride = [16, ]
        (self.feed('data')
         .conv(3, 3, 32, 1, 1, name='conv_0')
         .separable_conv(3, 3, 32, 1, 1, 1, name='conv1_sep')
         .separable_conv(3, 3, 32, 1, 1, 1, name='conv2_sep')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .separable_conv(3, 3, 64, 1, 1, 1, name='conv3_sep')
         .separable_conv(3, 3, 64, 1, 1, 1, name='conv4_sep')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
         .separable_conv(3, 3, 128, 1, 1, 1, name='conv5_sep')
         .separable_conv(3, 3, 128, 1, 1, 1, name='conv6_sep')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
         .separable_conv(3, 3, 256, 1, 1, 1, name='conv7_sep')
         .separable_conv(3, 3, 256, 1, 1, 1, name='conv8_sep')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
         .separable_conv(3, 3, 512, 1, 1, 1, name='conv9_sep')
         .separable_conv(3, 3, 512, 1, 1, 1, name='conv10_sep')
         .conv(3, 3, 512, 1, 1, name='rpn_conv/3x3'))  # RPN

        (self.feed('rpn_conv/3x3').Bilstm(512, 128, 512, name='lstm_o'))
        (self.feed('lstm_o').lstm_fc(512, len(anchor_scales) * 10 * 4, name='rpn_bbox_pred'))
        (self.feed('lstm_o').lstm_fc(512, len(anchor_scales) * 10 * 2, name='rpn_cls_score'))

        #  shape is (1, H, W, Ax2) -> (1, H, WxA, 2)
        (self.feed('rpn_cls_score')
         .spatial_reshape_layer(2, name='rpn_cls_score_reshape')
         .spatial_softmax(name='rpn_cls_prob'))

        # shape is (1, H, WxA, 2) -> (1, H, W, Ax2)
        (self.feed('rpn_cls_prob')
         .spatial_reshape_layer(len(anchor_scales) * 10 * 2, name='rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
         .proposal_layer(_feat_stride, anchor_scales, 'TEST', name='rois'))
