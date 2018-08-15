import os
import tensorflow as tf
import numpy as np
from ..utils.text_encode import LabelConverter
from ..utils.timer import Timer
from .config import cfg_from_file, cfg, get_log_dir
from ..networks.data_layer import TextDataSet
from ..networks.crnn import CRNN

_DEBUG = False
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cfg_from_file(os.path.join(root, 'crnn/text.yml'))
nclass = len(cfg.ALPHABET) + 1
imgH = 32
nh = 256
words = ['03JiuWuHi-Tech409132056', '82JiuWuHi-Tech011163106',
         '72JiuWuHi-Tech609143128', '62JiuWuHi-Tech209141048',
         'x2JiuWuHi-Tech105246174', '02JiuWuHi-Tech410216126',
         '42JiuWuHi-Tech004102008', '72JiuWuHi-Tech109278305',
         '82JiuWuHi-Tech103167035', '12JiuWuHi-Tech210287068',
         '52JiuWuHi-Tech901267004', '32JiuWuHi-Tech801188235',
         '22JiuWuHi-Tech501315180', '32JiuWuHi-Tech702131126',
         '92JiuWuHi-Tech012097029', '12JiuWuHi-Tech312314066',
         'x2JiuWuHi-Tech806244170', '12JiuWuHi-Tech905066201',
         '72JiuWuHi-Tech606274181', '12JiuWuHi-Tech802133050']
converter = LabelConverter(alphabet=cfg.ALPHABET)
class SolverWrapper(object):
    def __init__(self):
        self.__flag = True  # 线程是否继续的标志
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.75
        self.log_dir = get_log_dir(cfg.NET_NAME)
        # create graph
        self.__graph = tf.Graph()
        # init session
        self.__sess = tf.Session(graph=self.__graph, config=config)
        with self.__sess.as_default():
            with self.__graph.as_default():
                self.__net = CRNN(imgH, nclass, nh)
                # For checkpoint
                self.saver = tf.train.Saver(max_to_keep=100, write_version=tf.train.SaverDef.V2)
                self.writer = tf.summary.FileWriter(logdir=self.log_dir, graph=self.__graph, flush_secs=5)

    def stop_iter(self):
        self.__flag = False
    def snapshot(self, iter, output_dir, print_out=print):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with self.__sess.as_default():
            with self.__graph.as_default():
                infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                         if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
                filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                            '_iter_{:d}'.format(iter+1) + '.ckpt')
                filename = os.path.join(output_dir, filename)

                self.saver.save(self.__sess, filename)
                print_out('Wrote snapshot to: {:s}'.format(filename))


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

    def build_loss(self, logits, labels, seq_len):
        '''ctc_loss'''
        with self.__sess.as_default():
            with self.__graph.as_default():
                loss = tf.nn.ctc_loss(inputs=logits,
                                      labels=labels,
                                      sequence_length=seq_len)
                cost = tf.reduce_mean(loss)
                return cost

    def val_model(self, accuracy, test_image, test_label, print_out=print):
        with self.__sess.as_default():
            with self.__graph.as_default():
                text_list = [words[test_label[i]] for i in range(len(test_label))]
                sparse_tensor, label_length = converter.encode_sparse_tensor(text_list)
                feed_dict = {
                    self.__net.data: test_image,
                    self.__net.gt_text: sparse_tensor
                }
                timer = Timer()
                timer.tic()
                accuracy_record = self.__sess.run(accuracy, feed_dict=feed_dict)
                _diff_time = timer.toc(average=False)
                print_out('validation accuracy: %.4f, speed: %.3fs / validation'
                          % (1-accuracy_record, _diff_time))

    def train_model(self, train_file, test_file, output_dir,
                    solver='Adam', learning_rate=0.01, batch_size=100,
                    snap_iter=100, max_iters=100000, restore=False,
                    print_out=print, plot_out=None):
        """Network training loop."""
        with self.__sess.as_default():
            with self.__graph.as_default():
                print_out('Output will be saved to `{:s}`'.format(output_dir))
                print_out('Logs will be saved to `{:s}`'.format(self.log_dir))
                # --transform data to TFrecords--
                print_out('transform data to TFrecords...')
                train_data = TextDataSet(filename=train_file,
                                         tf_file=os.path.join(root, 'data/train.tfreconrds'),
                                         batch_size=batch_size)
                test_data = TextDataSet(filename=test_file,
                                        tf_file=os.path.join(root, 'data/test.tfreconrds'),
                                        batch_size=batch_size)
                train_image, train_label = train_data.read_and_decode()
                test_image, test_label = test_data.read_and_decode()
                print_out('done!')
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=self.__sess, coord=coord)
                output = self.__net.get_output('lstm2')
                time_stamps = tf.shape(output)[0]
                seq_len = tf.multiply(tf.ones(cfg.TRAIN.BATCH_SIZE, dtype=tf.int32), time_stamps)
                ctc_loss = self.build_loss(logits=output, labels=self.__net.gt_text, seq_len=seq_len)
                decoded, log_prob = tf.nn.ctc_beam_search_decoder(output, seq_len, merge_repeated=False)
                accuracy = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), self.__net.gt_text))

                # scalar summary
                tf.summary.scalar('ctc_loss', ctc_loss)
                tf.summary.scalar('train_accuracy', 1-accuracy)

                summary_op = tf.summary.merge_all()

                # optimizer
                lr = tf.Variable(learning_rate, trainable=False)
                if solver == 'Adam':
                    opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.999)
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
                    grads, norm = tf.clip_by_global_norm(tf.gradients(ctc_loss, tvars), 10.0)
                    train_op = opt.apply_gradients(list(zip(grads, tvars)), global_step=global_step)
                else:
                    train_op = opt.minimize(ctc_loss, global_step=global_step)

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
                        restore_iter = int(stem.split('_')[-1])
                        self.__sess.run(global_step.assign(restore_iter))
                        print_out('done!')
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
                            print(lr)

                        # load data
                        image_train, label_train = self.__sess.run([train_image, train_label])
                        image_test, label_test = self.__sess.run([test_image, test_label])
                        text_list = [words[label_train[i]] for i in range(len(label_train))]
                        sparse_tensor, label_length = converter.encode_sparse_tensor(text_list)
                        feed_dict = {
                            self.__net.data: image_train,
                            self.__net.gt_text: sparse_tensor
                        }
                        res_fetches = []
                        fetch_list = [ctc_loss, accuracy, summary_op, train_op] + res_fetches

                        ctc_loss_val, accuracy_val, summary_str, _ = self.__sess.run(fetches=fetch_list, feed_dict=feed_dict)

                        self.writer.add_summary(summary=summary_str, global_step=global_step.eval())

                        _diff_time = timer.toc(average=False)


                        if (iter) % (cfg.TRAIN.DISPLAY) == 0:
                            print_out('iter: %d / %d, ctc_loss: %.4f, train_accuracy: %.4f, '
                                      'lr: %f speed: %.3fs / iter'
                                      % (iter, max_iters, ctc_loss_val, 1 - accuracy_val, lr.eval(), _diff_time))
                            plot_out(ctc_loss_val, iter)


                        if (iter+1) % snap_iter == 0:
                            last_snapshot_iter = iter
                            self.snapshot(iter, output_dir, print_out)
                            self.val_model(accuracy, image_test, label_test, print_out)
                            plot_out(ctc_loss_val, iter)
                    else:
                        break

                if last_snapshot_iter != iter:
                    self.snapshot(iter, output_dir, print_out)
                coord.request_stop()
                coord.join(threads)

    def closeSess(self):
        self.__sess.close()
