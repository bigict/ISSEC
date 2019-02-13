from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
import glob

from libs.visualization.summary_utils import visualize_input
from libs.datasets import preprocess
import libs.configs.config_v1 as cfg
FLAGS = tf.app.flags.FLAGS


def get_dataset(dataset_name, split_name, dataset_dir, 
        im_batch=1, is_training=False, file_pattern=None, reader=None):
    if file_pattern is None:
        file_pattern = dataset_name + '_' + split_name + '*.tfrecord' 

    tfrecords = glob.glob(dataset_dir + '/records/' + file_pattern)
    image, ih, iw, gt_boxes, gt_masks, num_instances, img_id = read(tfrecords)

    image, gt_boxes, gt_masks = preprocess.preprocess_image(image, gt_boxes, gt_masks, is_training)

    return image, ih, iw, gt_boxes, gt_masks, num_instances, img_id


def read(tfrecords_filename):

    if not isinstance(tfrecords_filename, list):
        tfrecords_filename = [tfrecords_filename]
    filename_queue = tf.train.string_input_producer(
        tfrecords_filename, num_epochs=None)

    options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/img_id': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'label/num_instances': tf.FixedLenFeature([], tf.int64),
            'label/gt_masks': tf.FixedLenFeature([], tf.string),
            'label/gt_boxes': tf.FixedLenFeature([], tf.string),
            'label/encoded': tf.FixedLenFeature([], tf.string),
            })
    img_id = features['image/img_id']
    ih = tf.cast(features['image/height'], tf.int32)
    iw = tf.cast(features['image/width'], tf.int32)
    num_instances = tf.cast(features['label/num_instances'], tf.int32)
    image = tf.decode_raw(features['image/encoded'], tf.float64)
    imsize = tf.size(image)
    image = tf.reshape(image, (ih, iw, FLAGS.input_channel))
    gt_boxes = tf.decode_raw(features['label/gt_boxes'], tf.float32)
    gt_boxes = tf.reshape(gt_boxes, [-1, 5])
    gt_masks = tf.decode_raw(features['label/gt_masks'], tf.uint8)
    gt_masks = tf.cast(gt_masks, tf.int32)
    gt_masks = tf.reshape(gt_masks, [num_instances, ih, iw])

    return image, ih, iw, gt_boxes, gt_masks, num_instances, img_id
