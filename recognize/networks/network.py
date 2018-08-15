# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from ..crnn.config import cfg

DEFAULT_PADDING = 'SAME'

def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs)==0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        elif len(self.inputs)==1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated
def batch_norm(inputs, is_training=True, is_conv_out=True,decay = 0.999, name=None):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])

        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, 0.001, name)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, 0.001, name)
class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, ignore_missing=False):
        data_dict = np.load(data_path,encoding='latin1').item()
        for key in data_dict:
            with tf.variable_scope(key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        session.run(var.assign(data_dict[key][subkey]))
                        print("assign pretrain model "+subkey+ " to "+key)
                    except ValueError:
                        print("ignore "+key)
                        if not ignore_missing:

                            raise

    def feed(self, *args):
        assert len(args)!=0
        self.inputs = []
        for layer in args:
            if isinstance(layer, str):
                try:
                    layer = self.layers[layer]
                    print(layer)
                except KeyError:
                    print(list(self.layers.keys()))
                    raise KeyError('Unknown layer name fed: %s'%layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print(list(self.layers.keys()))
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in list(self.layers.items()))+1
        return '%s_%d'%(prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True, regularizer=None):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')


    @layer
    def Bilstm(self, input, d_i, d_h, d_o, name, trainable=True):
        img = input
        with tf.variable_scope(name) as scope:
            shape = tf.shape(img)
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            img = tf.reshape(img, [N * H, W, C])
            img.set_shape([None, None, d_i])

            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(d_h, state_is_tuple=True)
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(d_h, state_is_tuple=True)

            lstm_out, last_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, img, dtype=tf.float32)
            lstm_out = tf.concat(lstm_out, axis=-1)

            lstm_out = tf.reshape(lstm_out, [N * H * W, 2*d_h])

            init_weights = tf.truncated_normal_initializer(stddev=0.1)
            init_biases = tf.constant_initializer(0.0)
            weights = self.make_var('weights', [2*d_h, d_o], init_weights, trainable, \
                                    regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [d_o], init_biases, trainable)
            outputs = tf.matmul(lstm_out, weights) + biases

            outputs = tf.reshape(outputs, [N, H, W, d_o])
            return outputs

    @layer
    def lstm(self, input, d_i,d_h,d_o, name, trainable=True):
        img = input
        with tf.variable_scope(name) as scope:
            shape = tf.shape(img)
            N,H,W,C = shape[0], shape[1],shape[2], shape[3]
            img = tf.reshape(img,[N*H,W,C])
            img.set_shape([None,None,d_i])

            lstm_cell = tf.contrib.rnn.LSTMCell(d_h, state_is_tuple=True)
            initial_state = lstm_cell.zero_state(N*H, dtype=tf.float32)

            lstm_out, last_state = tf.nn.dynamic_rnn(lstm_cell, img,
                                               initial_state=initial_state,dtype=tf.float32)

            lstm_out = tf.reshape(lstm_out,[N*H*W,d_h])


            init_weights = tf.truncated_normal_initializer(stddev=0.1)
            init_biases = tf.constant_initializer(0.0)
            weights = self.make_var('weights', [d_h, d_o], init_weights, trainable, \
                              regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [d_o], init_biases, trainable)
            outputs = tf.matmul(lstm_out, weights) + biases


            outputs = tf.reshape(outputs, [N,H,W,d_o])
            return outputs

    @layer
    def lstm_fc(self, input, d_i, d_o, name, trainable=True):
        with tf.variable_scope(name) as scope:
            shape = tf.shape(input)
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            input = tf.reshape(input, [N*H*W,C])

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [d_i, d_o], init_weights, trainable,
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [d_o], init_biases, trainable)

            _O = tf.matmul(input, kernel) + biases
            return tf.reshape(_O, [N, H, W, int(d_o)])

    @layer
    def conv(self, input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             biased=True,
             bn=True,
             relu=True,
             padding=DEFAULT_PADDING,
             trainable=True):
        """ contribution by miraclebiu, and biased option"""
        c_i = input.get_shape()[-1]
        convolve = lambda i, k: tf.nn.conv2d(input=i,
                                             filter=k,
                                             strides=[1, s_h, s_w, 1],
                                             padding=padding)
        with tf.variable_scope(name) as scope:
            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            conv = convolve(input, kernel)
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                conv = tf.nn.bias_add(conv, biases)
            if bn and relu:
                conv = tf.nn.relu(conv, name=scope.name)
                conv = batch_norm(conv, name=scope.name)
            elif relu:
                conv = tf.nn.relu(conv, name=scope.name)
            elif bn:
                conv = batch_norm(conv, name=scope.name)
            return conv

    def depthwise_conv(self, input,
                       k_h,
                       k_w,
                       ch_mult,
                       s_h,
                       s_w,
                       name=None,
                       biased=True,
                       bn=True,
                       relu=True,
                       padding=DEFAULT_PADDING,
                       trainable=True):
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        convolve = lambda i, k: tf.nn.depthwise_conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i, ch_mult], init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            conv = convolve(input, kernel)
            if biased:
                biases = self.make_var('biases', [c_i], init_biases, trainable)
                conv = tf.nn.bias_add(conv, biases)
            if bn and relu:
                conv = tf.nn.relu6(conv, name=scope.name)
                conv = batch_norm(conv, name=scope.name)
            elif relu:
                conv = tf.nn.relu6(conv, name=scope.name)
            elif bn:
                conv = batch_norm(conv, name=scope.name)
            return conv

    def pointwise_conv(self, input,
                       c_o,
                       name=None,
                       biased=True,
                       bn=True,
                       relu=True,
                       padding=DEFAULT_PADDING,
                       trainable=True):
        """ contribution by miraclebiu, and biased option"""
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        convolve = lambda i, k: tf.nn.conv2d(input=i,
                                             filter=k,
                                             strides=[1, 1, 1, 1],
                                             padding=padding)
        with tf.variable_scope(name) as scope:

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [1, 1, c_i, c_o], init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            conv = convolve(input, kernel)
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                conv = tf.nn.bias_add(conv, biases)
            if bn and relu:
                conv = tf.nn.relu6(conv, name=scope.name)
                conv = batch_norm(conv, name=scope.name)
            elif relu:
                conv = tf.nn.relu6(conv, name=scope.name)
            elif bn:
                conv = batch_norm(conv, name=scope.name)
            return conv

    @layer
    def separable_conv(self, input,
                       k_h,
                       k_w,
                       c_o,
                       ch_mult,
                       s_h,
                       s_w,
                       name,
                       biased=True,
                       bn=True,
                       relu=True,
                       padding=DEFAULT_PADDING,
                       trainable=True):
        '''
        Mobilenet v1
        '''
        dw_name = name + '_dw'
        depthwise = self.depthwise_conv(input,
                                        k_h=k_h,
                                        k_w=k_w,
                                        ch_mult=ch_mult,
                                        s_h=s_h,
                                        s_w=s_w,
                                        name=dw_name,
                                        biased=biased,
                                        bn=bn,
                                        relu=True,
                                        padding=padding,
                                        trainable=trainable)
        pw_name = name + '_pw'
        pointwise = self.pointwise_conv(depthwise,
                              c_o=c_o,
                              name=pw_name,
                              biased=biased,
                              bn=bn,
                              relu=relu,
                              padding='VALID',
                              trainable=trainable)
        return pointwise


    @layer
    def bottleneck_block(self, input,
                         k_h,
                         k_w,
                         c_o,
                         ch_mult,
                         s_h,
                         s_w,
                         expansion,
                         name,
                         biased=True,
                         bn=True,
                         relu=True,
                         padding=DEFAULT_PADDING,
                         trainable=True):
        '''
        Mobilenet v2
        '''
        c_i = input.get_shape()[-1]
        pw_in_name = name + '_pw_in'
        pointwise_in = self.pointwise_conv(input,
                                        c_o=c_i*expansion,
                                        name=pw_in_name,
                                        biased=biased,
                                        bn=bn,
                                        relu=relu,
                                        padding='VALID',
                                        trainable=trainable)
        dw_name = name + '_dw'
        depthwise = self.depthwise_conv(pointwise_in,
                                        k_h=k_h,
                                        k_w=k_w,
                                        ch_mult=ch_mult,
                                        s_h=s_h,
                                        s_w=s_w,
                                        name=dw_name,
                                        biased=biased,
                                        bn=bn,
                                        relu=True,
                                        padding=padding,
                                        trainable=trainable)
        pw_out_name = name + '_pw_out'
        pointwise_out = self.pointwise_conv(depthwise,
                                           c_o=c_o,
                                           name=pw_out_name,
                                           biased=biased,
                                           bn=bn,
                                           relu=False,
                                           padding='VALID',
                                           trainable=trainable)

        residual = self.pointwise_conv(input,
                                       c_o=c_o,
                                       name=name+'residual',
                                       biased=biased,
                                       bn=True,
                                       relu=False,
                                       padding='VALID',
                                       trainable=trainable)
        return tf.nn.relu(tf.add(pointwise_out, residual))




    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def relu6(self, input, name):
        return tf.nn.relu6(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob_reshape':
            #
            # transpose: (1, AxH, W, 2) -> (1, 2, AxH, W)
            # reshape: (1, 2xA, H, W)
            # transpose: -> (1, H, W, 2xA)
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),
                                            [   input_shape[0],
                                                int(d),
                                                tf.cast(tf.cast(input_shape[1],tf.float32)/tf.cast(d,tf.float32)*tf.cast(input_shape[3],tf.float32),tf.int32),
                                                input_shape[2]
                                            ]),
                                 [0,2,3,1],name=name)
        else:
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),
                                        [   input_shape[0],
                                            int(d),
                                            tf.cast(tf.cast(input_shape[1],tf.float32)*(tf.cast(input_shape[3],tf.float32)/tf.cast(d,tf.float32)),tf.int32),
                                            input_shape[2]
                                        ]),
                                 [0,2,3,1],name=name)

    @layer
    def spatial_reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)
        # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
        return tf.reshape(input,\
                               [input_shape[0],\
                                input_shape[1], \
                                -1,\
                                int(d)])


    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(concat_dim=axis, values=inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True, trainable=True):
        with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input,[0,3,1,2]), [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
                init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [dim, num_out], init_weights, trainable, \
                                    regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            return tf.reshape(tf.nn.softmax(tf.reshape(input,[-1,input_shape[3]])),[-1,input_shape[1],input_shape[2],input_shape[3]],name=name)
        else:
            return tf.nn.softmax(input,name=name)

    @layer
    def spatial_softmax(self, input, name):
        input_shape = tf.shape(input)
        # d = input.get_shape()[-1]
        return tf.reshape(tf.nn.softmax(tf.reshape(input, [-1, input_shape[3]])),
                          [-1, input_shape[1], input_shape[2], input_shape[3]], name=name)

    @layer
    def add(self,input,name):
        """contribution by miraclebiu"""
        return tf.add(input[0],input[1])

    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)

    def l2_regularizer(self, weight_decay=0.0005, scope=None):
        def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                l2_weight = tf.convert_to_tensor(weight_decay,
                                       dtype=tensor.dtype.base_dtype,
                                       name='weight_decay')
                #return tf.mul(l2_weight, tf.nn.l2_loss(tensor), name='value')
                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
        return regularizer

    def smooth_l1_dist(self, deltas, sigma2=9.0, name='smooth_l1_dist'):
        with tf.name_scope(name=name) as scope:
            deltas_abs = tf.abs(deltas)
            smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0/sigma2), tf.float32)
            return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
                        (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)








