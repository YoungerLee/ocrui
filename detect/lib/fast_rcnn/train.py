from __future__ import print_function
import numpy as np
import os
import tensorflow as tf
from ..roi_data_layer.layer import RoIDataLayer
from ..utils.timer import Timer
from ..roi_data_layer import roidb as rdl_roidb
from ..fast_rcnn.config import cfg
from ..networks.factory import get_network
from .config import cfg_from_file, get_output_dir, get_log_dir, cfg
from ..datasets.factory import get_imdb

_DEBUG = False
# # 单例模式
# def Singleton(cls):
#     _instance = {}
#
#     def _singleton(*args, **kargs):
#         if cls not in _instance:
#             _instance[cls] = cls(*args, **kargs)
#         return _instance[cls]
#
#     return _singleton
# # solver只能初始化一次
# @Singleton
class SolverWrapper(object):
    def __init__(self):
        """Initialize the SolverWrapper."""
        # 线程是否继续的标志
        self.__flag = True
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.75
        # create graph
        self.__graph = tf.Graph()
        # init session
        self.__sess = tf.Session(graph=self.__graph, config=config)
        with self.__sess.as_default():
            with self.__graph.as_default():
                # load network
                self.__net = get_network("Mobilenet_train")
                cfg_from_file('./detect/ctpn/text.yml')
                imdb = get_imdb('voc_2007_trainval')
                roidb = get_training_roidb(imdb)
                self.log_dir = get_log_dir(imdb)
                self.imdb = imdb
                self.roidb = roidb

                print('Computing bounding-box regression targets...')
                if cfg.TRAIN.BBOX_REG:
                    self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
                print('done')

                # For checkpoint
                self.saver = tf.train.Saver(max_to_keep=100, write_version=tf.train.SaverDef.V2)
                self.writer = tf.summary.FileWriter(logdir=self.log_dir, graph=self.__graph, flush_secs=5)

    def stop_iter(self):
        self.__flag = False

    def snapshot(self, iter, output_dir, print_out=print):
        with self.__sess.as_default():
            with self.__graph.as_default():
                net = self.__net
                if cfg.TRAIN.BBOX_REG and 'bbox_pred' in net.layers and cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
                    # save original values
                    with tf.variable_scope('bbox_pred', reuse=True):
                        weights = tf.get_variable("weights")
                        biases = tf.get_variable("biases")

                    orig_0 = weights.eval()
                    orig_1 = biases.eval()

                    # scale and shift with bbox reg unnormalization; then save snapshot
                    weights_shape = weights.get_shape().as_list()
                    self.__sess.run(weights.assign(orig_0 * np.tile(self.bbox_stds, (weights_shape[0],1))))
                    self.__sess.run(biases.assign(orig_1 * self.bbox_stds + self.bbox_means))

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                         if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
                filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                            '_iter_{:d}'.format(iter+1) + '.ckpt')
                filename = os.path.join(output_dir, filename)

                self.saver.save(self.__sess, filename)
                print_out('Wrote snapshot to: {:s}'.format(filename))

                if cfg.TRAIN.BBOX_REG and 'bbox_pred' in net.layers:
                    # restore net to original state
                    self.__sess.run(weights.assign(orig_0))
                    self.__sess.run(biases.assign(orig_1))

    def build_image_summary(self):
        # A simple graph for write image summary
        with self.__sess.as_default():
            with self.__graph.as_default():
                log_image_data = tf.placeholder(tf.uint8, [None, None, 3])
                log_image_name = tf.placeholder(tf.string)
                # import tensorflow.python.ops.gen_logging_ops as logging_ops
                from tensorflow.python.ops import gen_logging_ops
                from tensorflow.python.framework import ops as _ops
                log_image = gen_logging_ops.image_summary(log_image_name, tf.expand_dims(log_image_data, 0), max_images=1)
                _ops.add_to_collection(_ops.GraphKeys.SUMMARIES, log_image)
                # log_image = tf.summary.image(log_image_name, tf.expand_dims(log_image_data, 0), max_outputs=1)
                return log_image, log_image_data, log_image_name


    def train_model(self, output_dir, solver='Adam', learning_rate=0.0001,
                    batch_size=300, snap_iter=100, max_iters=100000,
                    restore=True, print_out=print, plot_out=None):
        """Network training loop."""
        with self.__sess.as_default():
            with self.__graph.as_default():
                data_layer = get_data_layer(self.roidb, self.imdb.num_classes)
                total_loss, model_loss, rpn_cross_entropy, rpn_loss_box = self.__net.build_loss(ohem=cfg.TRAIN.OHEM)
                # scalar summary
                tf.summary.scalar('rpn_reg_loss', rpn_loss_box)
                tf.summary.scalar('rpn_cls_loss', rpn_cross_entropy)
                tf.summary.scalar('model_loss', model_loss)
                tf.summary.scalar('total_loss', total_loss)
                summary_op = tf.summary.merge_all()

                log_image, log_image_data, log_image_name =\
                    self.build_image_summary()

                # optimizer
                lr = tf.Variable(learning_rate, trainable=False)
                if solver == 'Adam':
                    opt = tf.train.AdamOptimizer(learning_rate)
                elif solver == 'RMS':
                    opt = tf.train.RMSPropOptimizer(learning_rate)
                else:
                    # lr = tf.Variable(0.0, trainable=False)
                    momentum = cfg.TRAIN.MOMENTUM
                    opt = tf.train.MomentumOptimizer(lr, momentum)

                global_step = tf.Variable(0, trainable=False)
                with_clip = True
                if with_clip:
                    tvars = tf.trainable_variables()
                    grads, norm = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), 10.0)
                    train_op = opt.apply_gradients(list(zip(grads, tvars)), global_step=global_step)
                else:
                    train_op = opt.minimize(total_loss, global_step=global_step)

                # intialize variables
                self.__sess.run(tf.global_variables_initializer())
                restore_iter = 0

                # resuming a trainer
                if restore:
                    try:
                        ckpt = tf.train.get_checkpoint_state(output_dir)
                        print_out('Restoring from {}...'.format(ckpt.model_checkpoint_path))
                        self.saver.restore(self.__sess, ckpt.model_checkpoint_path)
                        stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[0]
                        print_out(stem)
                        restore_iter = int(stem.split('_')[-1])
                        self.__sess.run(global_step.assign(restore_iter))
                        print_out('done')
                    except:
                        raise('Check your pretrained {:s}'.format(ckpt.model_checkpoint_path))

                last_snapshot_iter = -1
                timer = Timer()
                iter = 0
                for iter in range(restore_iter, max_iters):
                    if self.__flag:
                        timer.tic()
                        # learning rate
                        if iter != 0 and iter % cfg.TRAIN.STEPSIZE == 0:
                            self.__sess.run(tf.assign(lr, lr.eval() * cfg.TRAIN.GAMMA))
                            print_out(lr)

                        # get one batch
                        blobs = data_layer.forward()

                        feed_dict = {
                            self.__net.data: blobs['data'],
                            self.__net.im_info: blobs['im_info'],
                            self.__net.keep_prob: 0.5,
                            self.__net.gt_boxes: blobs['gt_boxes'],
                            self.__net.gt_ishard: blobs['gt_ishard'],
                            self.__net.dontcare_areas: blobs['dontcare_areas']
                        }
                        res_fetches = []
                        fetch_list = [total_loss,model_loss, rpn_cross_entropy, rpn_loss_box,
                                      summary_op,
                                      train_op] + res_fetches

                        total_loss_val, model_loss_val, rpn_loss_cls_val, rpn_loss_box_val, \
                            summary_str, _ = self.__sess.run(fetches=fetch_list, feed_dict=feed_dict)

                        self.writer.add_summary(summary=summary_str, global_step=global_step.eval())

                        _diff_time = timer.toc(average=False)

                        if iter % cfg.TRAIN.DISPLAY == 0:
                            print_out('iter: %d / %d, total loss: %.4f, model loss: %.4f, '
                                      'rpn_loss_cls: %.4f, rpn_loss_box: %.4f, lr: %f speed: %.4f'
                                      % (iter, max_iters, total_loss_val, model_loss_val, rpn_loss_cls_val,
                                         rpn_loss_box_val, lr.eval(), _diff_time))
                            plot_out(total_loss_val, iter)

                        if (iter+1) % snap_iter == 0:
                            last_snapshot_iter = iter
                            self.snapshot(iter, output_dir, print_out)
                    else:
                        break


                if last_snapshot_iter != iter:
                    self.snapshot(iter, output_dir, print_out)

    def closeSess(self):
        self.__sess.close()

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('done')

    print('Preparing training data...')
    if cfg.TRAIN.HAS_RPN:
            rdl_roidb.prepare_roidb(imdb)
    else:
        rdl_roidb.prepare_roidb(imdb)
    print('done')

    return imdb.roidb


def get_data_layer(roidb, num_classes):
    """return a data layer."""
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            # obsolete
            # layer = GtDataLayer(roidb)
            raise("Calling caffe modules...")
        else:
            layer = RoIDataLayer(roidb, num_classes)
    else:
        layer = RoIDataLayer(roidb, num_classes)

    return layer


