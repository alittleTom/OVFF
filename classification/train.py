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
import math

from input import Dataset
import globals as g_



currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', osp.dirname(sys.argv[0]) + './tmp/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('weights', '',
                            """finetune with a pretrained model""")
tf.app.flags.DEFINE_string('caffemodel','./alexnet_imagenet.npy',
                            """finetune with a model converted by caffe-tensorflow""")

np.set_printoptions(precision=3)


def train(dataset_train,dataset_test, ckptfile='', caffemodel=''):
    print('train() called')
    is_finetune = bool(ckptfile)
    V = g_.NUM_VIEWS
    batch_size = FLAGS.batch_size

    dataset_train.shuffle()
    data_size = dataset_train.size()
    print('training size:', data_size)


    with tf.Graph().as_default():
        startstep = 0 if not is_finetune else int(ckptfile.split('-')[-1])
        global_step = tf.Variable(startstep, trainable=False)
         
        # placeholders for graph input
        view_ = tf.placeholder('float32', shape=(None, V, 227, 227, 3), name='im0')
        y_ = tf.placeholder('int64', shape=(None), name='y')
        keep_prob_ = tf.placeholder('float32')

        # graph outputs
        fc8 = model.inference_multiview(view_, g_.NUM_CLASSES, keep_prob_)
        loss = model.loss(fc8, y_)
        train_op = model.train(loss, global_step, data_size)
        prediction = model.classify(fc8)

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1000)

        init_op = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
        
        if is_finetune:
            # load checkpoint file
            saver.restore(sess, ckptfile)
            print('restore variables done')
        elif caffemodel:
            # load caffemodel generated with caffe-tensorflow
            sess.run(init_op)
            model.load_alexnet_to_mvcnn(sess, caffemodel)
            print('loaded pretrained caffemodel:', caffemodel)
        else:
            # from scratch
            sess.run(init_op)
            print('init_op done')

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                               graph=sess.graph) 

        step = startstep
        epoch_num = 401  #301
        _loss_log = []
        _test_acc = []
        _test_log = []
        for epoch in range(epoch_num):
            print('epoch:', epoch)
            _loss = []
            for batch_x, batch_y in dataset_train.batches(batch_size):

                step += 1

                start_time = time.time()
                feed_dict = {view_: batch_x,
                             y_ : batch_y,
                             keep_prob_: 0.5 }

                _, pred, loss_value = sess.run(
                        [train_op, prediction,  loss,],
                        feed_dict=feed_dict)


                duration = time.time() - start_time
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                _loss.append(loss_value)
                # print training information
                if step % 10 == 0 or step - startstep <= 30:
                    sec_per_batch = float(duration)
                    print('%s: step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)' \
                         % (datetime.now(), step, loss_value,
                                    FLAGS.batch_size/duration, sec_per_batch))

                    acc = metrics.accuracy_score(batch_y, pred)
                    print('acc', acc * 100.)
            _loss_v = np.mean(np.array(_loss))
            _loss_log.append(_loss_v)
            
            startstep = 0
            step = startstep
            predictions = []
            labels = []
            data_size = dataset_test.size()

            print("Start testing")
            print("Size:", data_size)
            print("It'll take", int(math.ceil(data_size/batch_size)), "iterations.")
            _loss = []

            for batch_x, batch_y in dataset_test.batches(batch_size):
                step += 1

                start_time = time.time()
                feed_dict = {view_: batch_x,
                            y_ : batch_y,
                            keep_prob_: 1.0}

                pred, loss_value = sess.run(
                        [prediction,  loss,],
                        feed_dict=feed_dict)


                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                _loss.append(loss_value)

                predictions.extend(pred.tolist())
                labels.extend(batch_y.tolist())
            _loss_v = np.mean(np.array(_loss))
            _test_log.append(_loss_v)
            acc = metrics.accuracy_score(labels, predictions)
            print('acc:', acc * 100)
            _test_acc.append(acc)
            

        np.save('loss_log.npy',np.array(_loss_log))
        np.save('test_acc.npy',np.array(_test_acc))
        np.save('test_log.npy',np.array(_test_log))

        



def main(argv):
    st = time.time()
    print('start loading data')
    listfiles_train, labels_train = read_lists(g_.TRAIN_LOL)
    dataset_train = Dataset(listfiles_train, labels_train, subtract_mean=False, V=g_.NUM_VIEWS)
    print('done loading data, time=', time.time() - st)
    listfiles_test, labels_test = read_lists(g_.TEST_LOL)
    dataset_test = Dataset(listfiles_test, labels_test, subtract_mean=False, V=g_.NUM_VIEWS)

    train_time = time.time()
    train(dataset_train,dataset_test, FLAGS.weights, FLAGS.caffemodel)
    print('train time=',time.time()-train_time)


def read_lists(list_of_lists_file):
    #listfile_labels = np.loadtxt(list_of_lists_file, dtype=str).tolist()
    #listfiles, labels  = zip(*[(l[0], int(l[1])) for l in listfile_labels])
    #return listfiles, labels
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

