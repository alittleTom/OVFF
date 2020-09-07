import numpy as np
import os,sys,inspect
import tensorflow as tf
import time
from datetime import datetime
import os
import os.path as osp
from glob import glob
import sklearn.metrics as metrics
import fileinput

from input import Dataset
from input import BatchGenerator
import globals as g_

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import model


def train(dataset_train, ckptfile='', caffemodel=''):
    print('train() called')
    is_finetune = bool(ckptfile)
    V = g_.NUM_VIEWS

    with tf.Graph().as_default():
        startstep = 0 if not is_finetune else int(ckptfile.split('-')[-1])
        global_step = tf.Variable(startstep, trainable=False)
         
        # placeholders for graph input
        view_1_ = tf.placeholder('float32', shape=(None, V, 227, 227, 3), name='view1')
        view_2_ = tf.placeholder('float32', shape=(None, V, 227, 227, 3), name='view2')
        y_ = tf.placeholder('float32', shape=(None), name='y')
        keep_prob_ = tf.placeholder('float32')

        # graph outputs
        with tf.variable_scope('siemase') as scope:
            fc8_1_ = model.inference_multiview(view_1_, g_.DIM_DISCRIPTOR, keep_prob_)
            fc8_2_ = model.inference_multiview1(view_2_, g_.DIM_DISCRIPTOR, keep_prob_)

        loss = model.loss_with_spring(fc8_1_, fc8_2_, y_)
        train_op = model.train(loss, global_step, dataset_train.size())

        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=30)

        init_op = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=g_.log_device_placement))
        
        if is_finetune:
            # load checkpoint file
            saver.restore(sess, ckptfile)
            print('restore variables done')
        elif caffemodel:
            # load caffemodel generated with caffe-tensorflow
            sess.run(init_op)
            with tf.variable_scope('siemase', reuse=True) as scope:
                model.load_alexnet_to_mvcnn(sess, caffemodel)
                print('loaded pretrained caffemodel:', caffemodel)
        else:
            # from scratch
            sess.run(init_op)
            print('init_op done')

        summary_writer = tf.summary.FileWriter(g_.train_dir, graph=sess.graph)

        step = 0
        for step in range(1, 3001):
            [batch_x1, batch_x2, batch_y] = dataset_train.next_batch()
            print('batch_y', batch_y)

            feed_dict = {view_1_: batch_x1,
                            view_2_: batch_x2,
                            y_ : batch_y,
                            keep_prob_: 0.5 }

            _, loss_value = sess.run(
                          [train_op, loss], feed_dict=feed_dict)

            print('step: ', step, 'loss:', loss_value)

            if step % 3000 == 0:
                save_path = saver.save(sess, './output/model.ckpt', global_step=step)
                print("Model saved in file: %s" % save_path)


def main(argv):
    st = time.time()
    print('start loading data')
    listfiles_train, labels_train = read_lists(g_.TRAIN_LOL)
    dataset_train = BatchGenerator(listfiles_train, labels_train, subtract_mean=False, V=g_.NUM_VIEWS)
    print('done loading data, time=', time.time() - st)

    st2 = time.time()
    train(dataset_train, g_.weights, g_.caffemodel)
    print('train data, time=', time.time() - st2)


def read_lists(list_of_lists_file):
    listfiles = []
    labels = []
    with fileinput.input(files = (list_of_lists_file)) as f:
        for line in f:
            a,b = line.split()
            listfiles.append(a)
            labels.append(int(b))
    return listfiles, labels
    

if __name__ == '__main__':
    main(sys.argv)

