import tensorflow as tf
import os
from PIL import Image
import numpy as np
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class TextDataSet(object):
    def __init__(self, filename,
                 tf_file,
                 image_width=512,
                 image_height=32,
                 batch_size=100):
        self.filename = filename
        self.tf_file = tf_file
        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size
        image_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            lines = [line.strip('\n') for line in lines]
        for line in lines:
            img_file = line.split(' ')[0]

            image_list.append(img_file)
        self.image_list = image_list
        self._convert_to_tfrecords()
    def _convert_to_tfrecords(self):
        writer = tf.python_io.TFRecordWriter(self.tf_file)
        for img_path in self.image_list:
            try:
                img = Image.open(img_path).convert('L')
                label = int(img_path.split('/')[-2])
                img = img.resize((self.image_width, self.image_height))
                img_raw = img.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': int64_feature(label),
                    'image': bytes_feature(img_raw)
                }))
                writer.write(example.SerializeToString())
            except IOError as e:
                print('Could not read: ', img_path)
        writer.close()
        print('Transform done!')

    def read_and_decode(self):
        filename_queue = tf.train.string_input_producer([self.tf_file])
        reader = tf.TFRecordReader()
        _, serilized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serilized_example, features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string)
        })
        image = tf.decode_raw(features['image'], tf.uint8)
        image = tf.reshape(image, [self.image_height, self.image_width, 1])
        image = tf.cast(image, tf.float32) / 127.5 - 1.0
        # #--图像预处理--
        # # 随机裁剪
        # image = tf.image.resize_image_with_crop_or_pad(image, self.image_height, self.image_width)
        # # 随机水平、垂直翻转
        # image = tf.image.random_flip_left_right(image)
        # image = tf.image.random_flip_up_down(image)
        # # 随机调整亮度
        # image = tf.image.random_brightness(image, max_delta=63)
        # # 随机调整对比度
        # image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        # # 对图像进行白化操作，即像素值转为零均值单位方差
        # image = tf.image.per_image_standardization(image)

        label = tf.cast(features['label'], tf.int32)

        image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                          batch_size=self.batch_size,
                                                          min_after_dequeue=100,
                                                          num_threads=64,
                                                          capacity=100+(64+4)*self.batch_size)
        return image_batch, label_batch






    # def get_batch(self, batch_size=100):
    #     image = tf.convert_to_tensor(self.image_list, tf.string)
    #     label = tf.convert_to_tensor(self.label_list, tf.int32)
    #     imagepath, label = tf.train.slice_input_producer([image, label])
    #     image_contents = tf.read_file(imagepath)
    #     image = tf.image.decode_jpeg(image_contents, channels=1) # read as grayscale
    #     # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    #     image = tf.image.resize_images(image, size=[IMAGE_HEIGHT, IMAGE_WIDTH])
    #     image = image * 1.0 / 127.5 - 1.0
    #     #--图像预处理--
    #     # 统一大小
    #     # image = tf.image.resize_images(image, size=[IMAGE_HEIGHT, IMAGE_WIDTH])
    #     # # 随机裁剪
    #     # # image = tf.image.resize_image_with_crop_or_pad(image, img_width, img_height)
    #     # # 随机水平、垂直翻转
    #     # image = tf.image.random_flip_left_right(image)
    #     # image = tf.image.random_flip_up_down(image)
    #     # # 随机调整亮度
    #     # image = tf.image.random_brightness(image, max_delta=63)
    #     # # 随机调整对比度
    #     # image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    #     # # 对图像进行白化操作，即像素值转为零均值单位方差
    #     # image = tf.image.per_image_standardization(image)
    #     image_batch, label_batch = tf.train.batch([image, label],
    #                                                       batch_size=batch_size,
    #                                                       num_threads=64,
    #                                                       capacity=capacity)
    #     return image_batch, label_batch

