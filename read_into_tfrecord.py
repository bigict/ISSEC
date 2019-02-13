#!/usr/bin/env python

import sys
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType

from utils.util import LoadNativeBox
import libs.configs.config_v1 as cfg

FLAGS = tf.app.flags.FLAGS

import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'

def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def to_tfexample_raw(image_id, image_data, label_data, height, width,
                    num_instances, gt_boxes, masks):
    """ just write a raw input"""
    return tf.train.Example(features=tf.train.Features(feature={
        'image/img_id': _bytes_feature(image_id),
        'image/encoded': _bytes_feature(image_data),
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'label/num_instances': _int64_feature(num_instances),    # N
        'label/gt_boxes': _bytes_feature(gt_boxes),    # of shape (N, 5), (x1, y1, x2, y2, classid)
        'label/gt_masks': _bytes_feature(masks),    # of shape (N, height, width)
        'label/encoded': _bytes_feature(label_data),    # deprecated, this is used for pixel-level segmentation
    }))

def add_to_tfrecord():
    """Loads image files and writes files to a TFRecord.
    Note: masks and bboxes will lose shape info after converting to string.
    """

    record_path = os.path.join(FLAGS.dataset_dir, "records")
    if not os.path.isdir(record_path):
        os.makedirs(record_path)

    ####### Path Definition #######
    img_path = "data/rawdata/ccmpred/"
    ss3_path = "data/rawdata/ss3/"
    pdb_path = "data/rawdata/pdb/"
    fasta_path = "data/rawdata/fasta/"

    with open('data/train.list') as fin:
        names = [line.rstrip() for line in fin]

    num_shards = int(len(names) / 1000)
    num_per_shard = int(math.ceil(len(names) / float(num_shards)))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            for shard_id in range(num_shards):
                record_filename = os.path.join(record_path, "new_train_ss3_%d.tfrecord" %shard_id)
                options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
                with tf.python_io.TFRecordWriter(record_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(names))
                    print "processing data from %d to %d..." %(start_ndx, end_ndx)
                    for i in range(start_ndx, end_ndx):
                        name = names[i]
                        img = np.loadtxt(os.path.join(img_path, name+".ccmpred"))
                        height, width = img.shape
                        # load ss3 info
                        predss3 = np.loadtxt(os.path.join(ss3_path, name+".ss3"))
                        if predss3.shape[0] != height:
                            continue
                        Index = np.mgrid[0: height, 0: width]
                        i, j = Index[0], Index[1]
                        ss3mat = np.concatenate([predss3[i], predss3[j]], axis=-1)
                        # combine ccmpred and ss3
                        img = np.concatenate([img[..., np.newaxis], ss3mat], axis=-1)

                        # box and mask info
                        pdb = os.path.join(pdb_path, name+".pdb")
                        fasta = os.path.join(fasta_path, name+".fasta")
                        L, gt_boxes, masks, ss3 = LoadNativeBox(name, pdb, fasta)
                        gt_boxes = np.array(gt_boxes).astype(np.float32)
                        masks = np.array(masks).astype(np.uint8)   ### Important ###
                        
                        # combine all masks in one mask
                        mask = np.zeros(shape=(height, width), dtype=np.int8)
                        for m in masks:
                            mask += m

                        img = img.astype(np.float64)
                        #assert img.size == width * height * 3, '%s' % str(name)
                        
                        #if gt_boxes.shape[0] > 0:
                        example = to_tfexample_raw(
                            name, img.tostring(), mask.tostring(),
                            height, width, 
                            gt_boxes.shape[0], gt_boxes.tostring(), masks.tostring())
                    
                        tfrecord_writer.write(example.SerializeToString())

def read():
    filename_queue = tf.train.string_input_producer(["6367proteins/records/6367proteins_test_00000-of-00001.tfrecord"], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
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
    image = tf.cond(tf.equal(imsize, ih * iw), \
                    lambda: tf.image.grayscale_to_rgb(tf.reshape(image, (ih, iw, 1))), \
                    lambda: tf.reshape(image, (ih, iw, 3)))

    gt_boxes = tf.decode_raw(features['label/gt_boxes'], tf.float32)
    gt_boxes = tf.reshape(gt_boxes, [num_instances, 5])
    gt_masks = tf.decode_raw(features['label/gt_masks'], tf.uint8)
    gt_masks = tf.cast(gt_masks, tf.int32)
    gt_masks = tf.reshape(gt_masks, [num_instances, ih, iw])

    sess = tf.Session()
    init = tf.initialize_local_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    name, image, h, w, n, m, box, mask = sess.run([img_id, image, ih, iw, num_instances, image, gt_boxes, gt_masks])
    #name = sess.run([img_id])
    #h = sess.run([ih])
    #w = sess.run([iw])
    #print w
    #m = sess.run([image])
    #box = sess.run([gt_boxes])
    #print box
    #mask = sess.run([gt_masks])
    print mask.shape

if __name__ == "__main__":
    if not os.path.isdir(FLAGS.dataset_dir):
        os.makedirs(FLAGS.dataset_dir)
    add_to_tfrecord()
