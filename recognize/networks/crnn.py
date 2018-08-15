import tensorflow as tf
from .network import Network

# backup
class CRNN(Network):
    def __init__(self, imgH, nclass, nh, trainable=True):
        self.inputs = []
        # init model size
        self.imgH = imgH
        self.nh = nh
        self.nclass = nclass
        # init placeholder
        self.data = tf.placeholder(tf.float32, shape=[None, 32, 512, 1], name='data')
        self.gt_text = tf.sparse_placeholder(tf.int32)
        self.layers = dict({'data': self.data})
        self.trainable = trainable
        self.setup()

    def setup(self):
        # (self.feed('data')
        #  .conv(3, 3, 64, 1, 1, name='conv0')
        #  .max_pool(2, 2, 2, 2, padding='VALID', name='pool0')
        #  .separable_conv(3, 3, 128, 1, 1, 1, name='conv1')
        #  .separable_conv(3, 3, 128, 1, 1, 1, name='conv2')
        #  .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
        #  .separable_conv(3, 3, 256, 1, 1, 1, name='conv3')
        #  .separable_conv(3, 3, 256, 1, 1, 1, name='conv4')
        #  .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
        #  .separable_conv(3, 3, 512, 1, 1, 1, name='conv5')
        #  .separable_conv(3, 3, 512, 1, 1, 1, name='conv6')
        #  .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
        #  .conv(2, 1, 512, 1, 1, padding='VALID', name='conv_o')
        #  )
        (self.feed('data')
         .conv(3, 3, 64, 1, 1, name='conv0')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool0')
         .bottleneck_block(3, 3, 16, 1, 1, 1, expansion=1, name='block1')
         .bottleneck_block(3, 3, 24, 1, 1, 1, expansion=6, name='block2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .bottleneck_block(3, 3, 32, 1, 1, 1, expansion=6, name='block3')
         .bottleneck_block(3, 3, 64, 1, 1, 1, expansion=6, name='block4')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
         .bottleneck_block(3, 3, 128, 1, 1, 1, expansion=6, name='block5')
         .bottleneck_block(3, 3, 256, 1, 1, 1, expansion=6, name='block6')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
         .conv(2, 1, 512, 1, 1, padding='VALID', name='conv_o')
         )

        (self.feed('conv_o')
         .Bilstm(512, self.nh, self.nh, name='lstm1')
         .Bilstm(self.nh, self.nh, self.nclass, name='lstm2'))
        output = self.get_output('lstm2')
        shape = tf.shape(output)
        N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        output = tf.reshape(output, [N*H, -1, C])
        self.layers['lstm2'] = tf.transpose(output, perm=[1, 0, 2])
        # self.layers['lstm_o'] = tf.log(tf.transpose(output, perm=[1, 0, 2]) + EPSILON)
