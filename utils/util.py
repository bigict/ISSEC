import tensorflow as tf
import numpy as np

from utils.proteinINFO import proteinINFO
from utils.matrix4sse import matrix4sse

import libs.preprocessings.utils as preprocess_utils
from libs.visualization import visualize
import libs.configs.config_v1 as cfg
from libs.nms.py_cpu_nms import py_cpu_nms

FLAGS = tf.app.flags.FLAGS

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

def LoadNativeBox(name, pdbfile, fastafile):
    proinfo = proteinINFO(name, pdbfile, fastafile)
    ss3 = proinfo.ss3seq
    dist_matrix = proinfo.dist_matrix
    angle_matrix = proinfo.angle_matrix
    c = matrix4sse(ss3, dist_matrix, angle_matrix)
    gt_boxes = c.boxes
    masks = c.masks
    ssenum = c.sse_matrix.shape[0]
    return ssenum, gt_boxes, masks, ss3

def preprocess_inputimage(image):
    """preprocess input image for network"""
    ih, iw = tf.shape(image)[0], tf.shape(image)[1]
    ## min size resizing
    new_ih, new_iw = preprocess_utils._smallest_size_at_least(ih, iw, cfg.FLAGS.image_min_size)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [new_ih, new_iw], align_corners=False)
    image = tf.squeeze(image, axis=[0])
    ## zero mean image
    image = tf.cast(image, tf.float32)
    image = image / 256.0
    image = (image - 0.5) * 2.0
    image = tf.expand_dims(image, axis=0)
    ## rgb to bgr
    image = tf.reverse(image, axis=[-1])
    return image

def process_output(final_boxnp, final_clsnp, final_masknp, final_probnp, img_shape, 
        score_threshold = 0.6, cls = -1, nms_threshold = 0.1):
    # score_threshold: instance with score > score_threshold was a positive prediction
    # cls: -1 refer to all 3 types, 0 refer to helix-helix, 1(2) refer to beta-beta parallel(anti-parallel)
    pd_boxes = np.concatenate((final_boxnp, final_clsnp[..., np.newaxis]), axis=-1)
    pd_boxes = pd_boxes.astype(np.int32).tolist()
    pd_masks = final_masknp
    cc_boxes = []
    # exclude background boxes
    for i in xrange(len(pd_boxes)):
        if cls == -1:
            if pd_boxes[i][-1] == 0:
                continue
            class_id = pd_boxes[i][-1]
            score = final_probnp[i][class_id]
            cc_boxes.append(pd_boxes[i] + [pd_masks[i][:,:,class_id]] + [score])
            # adding symmetrical boxes
            cc_boxes.append([pd_boxes[i][1],pd_boxes[i][0],pd_boxes[i][3],
                             pd_boxes[i][2],pd_boxes[i][-1],pd_masks[i][:,:,class_id],score])
        elif pd_boxes[i][-1] == cls:
            class_id = pd_boxes[i][-1]
            score = final_probnp[i][class_id]
            cc_boxes.append(pd_boxes[i] + [pd_masks[i][:,:,class_id]] + [score])
            # adding symmetrical boxes
            cc_boxes.append([pd_boxes[i][1],pd_boxes[i][0],pd_boxes[i][3],
                             pd_boxes[i][2],pd_boxes[i][-1],pd_masks[i][:,:,class_id],score])
    pred_boxes = []
    pred_class = []
    scores = []
    pred_masks = []
    # selected with NMS
    #filter_boxes = sorted(cc_boxes, key=lambda x: x[-1], reverse=True)
    dets = []
    for b in cc_boxes:
        dets.append([b[0], b[1], b[2], b[3], b[-1]])
    if len(dets) == 0:
        keep = []
    else:
        keep = py_cpu_nms(np.array(dets), thresh = nms_threshold)
    for pos in keep:
        p = cc_boxes[pos]
        score = p[-1]
        if score < score_threshold:
            continue
        box = p[:4]
        class_id = p[4]
        mask = p[-2]
        pred_mask = visualize.unmold_mask(mask, box, img_shape)
        pred_boxes.append(box)
        pred_class.append(class_id)
        scores.append(score)
        pred_masks.append(pred_mask)

    return pred_boxes, pred_masks, pred_class, scores
