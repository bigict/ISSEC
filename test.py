import tensorflow as tf
import numpy as np
import argparse
import sys

from utils.proteinINFO import proteinINFO
from utils.matrix4sse import matrix4sse

import libs.preprocessings.utils as preprocess_utils
import libs.configs.config_v1 as cfg
import libs.nets.nets_factory as network 
import libs.nets.pyramid_network as pyramid_network
from libs.nms.py_cpu_nms import py_cpu_nms
from libs.visualization import visualize

from utils.util import preprocess_inputimage, LoadNativeBox, process_output
from utils.eval import prf


import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('m', '', 'kernel')
tf.app.flags.DEFINE_string('d', '', 'kernel')
tf.app.flags.DEFINE_string('s', '', 'kernel')

datasetlist = ['psicov', 'mem30', 'mem11', 'BetaSheet186']

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for testing model", usage='%(prog)s <-m model_path> <-d dataset> [options]', add_help=False)
    
    input_args = parser.add_argument_group('Input arguments')
    input_args.add_argument('-m', '--model', metavar='<model_path>', help='Input model path')
    input_args.add_argument('-d', '--dataset', metavar='<dataset>', help="Name should be in "+"("+", ".join(datasetlist)+")")


    opt_args = parser.add_argument_group('Optional arguments')
    opt_args.add_argument('-s', '--singlename', metavar='[single_protein_name]', type=open, help='if you want to test on single one protein')

    other_args = parser.add_argument_group('Other arguments')
    verbose_quiet = other_args.add_mutually_exclusive_group()
    verbose_quiet.add_argument('-q', '--quiet', action='store_true', help='be quiet, [default=True]')
    other_args.add_argument('-h', '--help', action='help', help='show this help message and exit')
    other_args.add_argument('--version', action='version', version='%(prog)s 1.0')

    return parser.parse_args()


def load_image(name, datasplit = "psicov", withss3 = True):
    img_path = os.path.join("data/testdata", datasplit, name+".ccmpred")
    ss3_path = os.path.join("data/testdata", datasplit, name+".ss3")
    img = np.loadtxt(img_path)
    if withss3:
        height, width = img.shape
        # load ss3 info
        ss3 = np.loadtxt(ss3_path)
        Index = np.mgrid[0: height, 0: width]
        i, j = Index[0], Index[1]
        ss3mat = np.concatenate([ss3[i], ss3[j]], axis=-1)
        # combine ccmpred and ss3
        img = np.concatenate([img[..., np.newaxis], ss3mat], axis=-1)
    image = img.astype(np.float32)
    return image

def main():

    args = parse_arguments()

    if not args.model:
        os.system("python " + sys.argv[0] + ' -h')
        sys.exit('\n\n\n!!!ERROR: -m (--model) model_path missing!!!\n\n\n')
    model_path = args.model

    if not args.dataset:
        os.system("python " + sys.argv[0] + ' -h')
        sys.exit('\n\n\n!!!ERROR: -d (--dataset) dataset name missing!!!\n\n\n')
    dataset = args.dataset

    if dataset not in datasetlist:
        sys.exit('\n\n\n!!!ERROR: dataset name not in [%s]!!!\n\n\n' %(", ".join(datasetlist)))

    with open(os.path.join("data/testdata/list/", dataset+".list")) as fin:
        names = [line.strip() for line in fin]

    if args.singlename:
        name = args.singlename
        if name not in names:
            sys.exit('\n\n\n!!!ERROR: %s not in %s dataset!!!\n\n\n' %(name, dataset))
        else:
            names = [name]

    # build graph
    test_image = tf.placeholder("float", shape=[None, None, None])
    ih = tf.shape(test_image)[0]
    iw = tf.shape(test_image)[1]
    image = tf.reshape(test_image, (ih, iw, FLAGS.input_channel))
    image = preprocess_inputimage(image)
    im_shape = tf.shape(image)
    image = tf.reshape(image, (im_shape[0], im_shape[1], im_shape[2], FLAGS.input_channel))

    logits, end_points, pyramid_map = network.get_network(FLAGS.network, image,
                        weight_decay=FLAGS.weight_decay, is_training=True)
    outputs = pyramid_network.build(end_points, im_shape[1], im_shape[2], pyramid_map,
                        num_classes=4,
                        base_anchors=9,
                        is_training=False,
                        gt_boxes=None, gt_masks=None,
                        loss_weights=[0.2, 0.2, 1.0, 0.2, 1.0])
    input_image = end_points['input']
    final_box = outputs['final_boxes']['box']
    final_cls = outputs['final_boxes']['cls']
    final_prob = outputs['final_boxes']['prob']
    final_mask = outputs['mask']['mask']

    #return boxes to origin shape
    scale_ratio = tf.cast(ih, tf.float32) / tf.cast(im_shape[2], tf.float32)
    final_box = final_box * scale_ratio

    ## restore
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #model_path = "../FastMaskRCNN/output/new_train_ss3/"
    print ("model path: %s" %model_path)
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    restorer = tf.train.Saver()
    restorer.restore(sess, checkpoint_path)
    
    # evaluation
    data_path = os.path.join("data/testdata", dataset)
    evalres = []
    for name in names:
        pdb = os.path.join(data_path, name+".pdb")
        fasta = os.path.join(data_path, name+".fasta")
        img = load_image(name, dataset)

        L, gt_boxes, true_masks, ss3 = LoadNativeBox(name, pdb, fasta)
        input_imagenp, final_boxnp, final_clsnp, final_masknp, final_probnp= \
                sess.run([input_image] + [final_box] + [final_cls] + [final_mask] + [final_prob], 
                feed_dict={test_image: img})

        pred_boxes, pred_masks, pred_classids, scores = \
                process_output(final_boxnp, final_clsnp, final_masknp, final_probnp, img_shape = img.shape[:2], 
                score_threshold = 0.7, cls = -1)
        precision, recall, F1 = prf(pred_boxes, gt_boxes)
        evalres.append([precision, recall, F1])
        print ("""%s: precision=%.3f, recall=%.3f, F1=%.3f""" %(name, precision, recall, F1))
    evalres = np.array(evalres)
    avg = np.mean(evalres, axis=0)
    print ("""\nAverage: precision=%.3f, recall=%.3f, F1=%.3f\n""" %(avg[0], avg[1], avg[2]))

if __name__ == "__main__":
    main()
