#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import functools
import os, sys
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from time import gmtime, strftime
import cPickle as pickle
import copy

import libs.configs.config_v1 as cfg
import libs.datasets.dataset_factory as datasets
import libs.nets.nets_factory as network 

import libs.nets.pyramid_network as pyramid_network
import libs.nets.resnet_v1 as resnet_v1

from utils.util import process_output
from utils.eval import prf 

from utils.train_utils import _configure_learning_rate, _configure_optimizer, \
  _get_variables_to_train, _get_init_fn, get_var_list_to_restore

from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from libs.visualization.pil_utils import cat_id_to_cls_name, draw_img, draw_bbox

# using GPU numbered 0
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'

FLAGS = tf.app.flags.FLAGS
resnet50 = resnet_v1.resnet_v1_50

def solve(global_step):
    """add solver to losses"""
    # learning reate
    lr = _configure_learning_rate(82783, global_step)
    optimizer = _configure_optimizer(lr)
    tf.summary.scalar('learning_rate', lr)

    # compute and apply gradient
    losses = tf.get_collection(tf.GraphKeys.LOSSES)
    regular_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regular_loss = tf.add_n(regular_losses)
    out_loss = tf.add_n(losses)
    total_loss = tf.add_n(losses + regular_losses)

    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('out_loss', out_loss)
    tf.summary.scalar('regular_loss', regular_loss)

    update_ops = []
    variables_to_train = _get_variables_to_train()
    # update_op = optimizer.minimize(total_loss)
    gradients = optimizer.compute_gradients(total_loss, var_list=variables_to_train)
    grad_updates = optimizer.apply_gradients(gradients, 
            global_step=global_step)
    update_ops.append(grad_updates)
    
    # update moving mean and variance
    if FLAGS.update_bn:
        update_bns = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_bn = tf.group(*update_bns)
        update_ops.append(update_bn)

    return tf.group(*update_ops)

def restore(sess):
     """choose which param to restore"""
     if FLAGS.restore_previous_if_exists:
        try:
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)
            restorer = tf.train.Saver()

            restorer.restore(sess, checkpoint_path)
            print ('restored previous model %s from %s'\
                    %(checkpoint_path, FLAGS.train_dir))
            time.sleep(2)
            return
        except:
            print ('--restore_previous_if_exists is set, but failed to restore in %s %s'\
                    % (FLAGS.train_dir, checkpoint_path))
            time.sleep(2)

def train():
    """The main function that runs training"""

    ## data
    image, ih, iw, gt_boxes, gt_masks, num_instances, img_id = \
        datasets.get_dataset(FLAGS.dataset_name, 
                             FLAGS.dataset_split_name, 
                             FLAGS.dataset_dir, 
                             FLAGS.im_batch,
                             is_training=True)
    data_queue = tf.RandomShuffleQueue(capacity=32, min_after_dequeue=16,
            dtypes=(
                image.dtype, ih.dtype, iw.dtype, 
                gt_boxes.dtype, gt_masks.dtype, 
                num_instances.dtype, img_id.dtype)) 
    enqueue_op = data_queue.enqueue((image, ih, iw, gt_boxes, gt_masks, num_instances, img_id))
    data_queue_runner = tf.train.QueueRunner(data_queue, [enqueue_op] * 4)
    tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, data_queue_runner)
    (image, ih, iw, gt_boxes, gt_masks, num_instances, img_id) = data_queue.dequeue()
    im_shape = tf.shape(image)
    image = tf.reshape(image, (im_shape[0], im_shape[1], im_shape[2], FLAGS.input_channel))

    ## network
    logits, end_points, pyramid_map = network.get_network(FLAGS.network, image,
            weight_decay=FLAGS.weight_decay, is_training=False)
    outputs = pyramid_network.build(end_points, im_shape[1], im_shape[2], pyramid_map,
            num_classes=4,
            base_anchors=9,
            is_training=True,
            gt_boxes=gt_boxes, gt_masks=gt_masks,
            loss_weights=[0.2, 0.2, 1.0, 0.2, 0.0])
            #default: loss_weights=[0.2, 0.2, 1.0, 0.2, 1.0])


    total_loss = outputs['total_loss']
    losses  = outputs['losses']
    batch_info = outputs['batch_info']
    regular_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    
    input_image = end_points['input']
    final_box = outputs['final_boxes']['box']
    final_cls = outputs['final_boxes']['cls']
    final_prob = outputs['final_boxes']['prob']
    final_mask = outputs['mask']['mask']

    #return boxes to origin shape
    scale_ratio = tf.cast(ih, tf.float32) / tf.cast(im_shape[2], tf.float32)
    final_box = final_box * scale_ratio
    gt_boxes = gt_boxes * scale_ratio

    ## solvers
    global_step = slim.create_global_step()
    update_op = solve(global_step)

    cropped_rois = tf.get_collection('__CROPPED__')[0]
    transposed = tf.get_collection('__TRANSPOSED__')[0]
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
            )
    sess.run(init_op)

    summary_op = tf.summary.merge_all()
    logdir = os.path.join(FLAGS.train_dir, strftime('%Y%m%d%H%M%S', gmtime()))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)

    ## restore
    restore(sess)

    ## main loop
    coord = tf.train.Coordinator()
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

    tf.train.start_queue_runners(sess=sess, coord=coord)
    saver = tf.train.Saver(max_to_keep=20)

    begin_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    for step in range(FLAGS.max_iters+1):
        
        start_time = time.time()

        s_, tot_loss, reg_lossnp, img_id_str, \
        gt_boxesnp, h, w, \
        input_imagenp, final_boxnp, final_clsnp, final_masknp, final_probnp = \
                sess.run([update_op, total_loss, regular_loss, img_id] +
                         [gt_boxes] + [ih] + [iw] +
                         [input_image] + [final_box] + [final_cls] + [final_mask] + [final_prob])

        duration_time = time.time() - start_time
        pred_boxes, pred_masks, pred_classids, scores = \
                    process_output(final_boxnp, final_clsnp, final_masknp, final_probnp, \
                    img_shape = (h, w), score_threshold = 0.7, cls = -1, nms_threshold = 0.1)
        precision, recall, F1 = prf(pred_boxes, gt_boxesnp.astype(np.int32).tolist())
        print ( """iter %d: image-id:%s, time:%.3f(sec), regular_loss: %.6f, total-loss %.4f, """
                """precision & recall & F1(%.2f, %.2f, %.2f)"""
               % (step, img_id_str, duration_time, reg_lossnp, tot_loss, precision, recall, F1))
        if step % 100 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

        if (step % 10000 == 0 or step + 1 == FLAGS.max_iters) and step != 0:
            checkpoint_path = os.path.join(FLAGS.train_dir, 
                                           FLAGS.dataset_name + '_' + FLAGS.network + '_model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

        if coord.should_stop():
            coord.request_stop()
            coord.join(threads)

    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("start-time: %s\nfinish-time: %s" %(begin_time, end_time))

if __name__ == '__main__':
    train()
