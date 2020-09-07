import numpy as np
import os,sys,inspect
import tensorflow as tf
import time
from datetime import datetime
import os
# import hickle as hkl
import os.path as osp
from glob import glob
# import sklearn.metrics as metrics
import math
import fileinput

from input import Dataset
import globals as g_


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import model


def test(dataset, ckptfile):
    print('test() called')
    V = g_.NUM_VIEWS
    batch_size = g_.batch_size

    data_size = dataset.size()
    print('dataset size:', data_size)

    with tf.Graph().as_default():
        startstep = 0
        global_step = tf.Variable(startstep, trainable=False)

        view_ = tf.placeholder('float32', shape=(None, V, 227, 227, 3), name='view')
        y_ = tf.placeholder('int64', shape=(None), name='y')
        keep_prob_ = tf.placeholder('float32')

        with tf.variable_scope('siemase') as scope:
            fc8_ = model.inference_multiview(view_, g_.DIM_DISCRIPTOR, keep_prob_)

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=g_.log_device_placement))
        saver.restore(sess, ckptfile)
        print('restore variables done')

        step = startstep
        descriptor = []
        labels = []

        print("Start testing")
        print("Size:", data_size)
        print("It'll take", int(math.ceil(data_size/batch_size)), "iterations.")


        for batch_x, batch_y in dataset.batches(batch_size):
            print(batch_x)
            print(batch_y)
            step += 1

            start_time = time.time()
            feed_dict = {view_: batch_x,
                         keep_prob_: 0.5}
            fc8_v = sess.run([fc8_], feed_dict=feed_dict)

            duration = time.time() - start_time

            descriptor.extend(fc8_v)
            labels.extend(batch_y)

        descriptor = np.array(descriptor)
        print(descriptor.shape)
        np.save('./output/descriptor.npy', descriptor)
        labels = np.array(labels)
        print(labels.shape)
        np.save('./output/labels.npy', labels)



def main(argv):
    st = time.time()
    print('start loading data')
    listfiles, labels = read_lists(g_.TEST_LOL)
    dataset = Dataset(listfiles, labels, subtract_mean=False, V=g_.NUM_VIEWS)
    print('done loading data, time=', time.time() - st)

    test(dataset, './output(64)/model.ckpt-3000')


def read_lists(list_of_lists_file):
    listfiles = []
    labels = []
    with fileinput.input(files=(list_of_lists_file)) as f:
        for line in f:
            a, b = line.split()
            listfiles.append(a)
            labels.append(int(b))
    return listfiles, labels


if __name__ == '__main__':
    main(sys.argv)


