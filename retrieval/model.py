#coding=utf-8
import tensorflow as tf
import re
import numpy as np
import globals as g_

def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % g_.TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)

    return var


def _variable_with_weight_decay(name, shape, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    var = _variable_on_cpu(name, shape,
                           initializer=tf.contrib.layers.xavier_initializer())
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _conv(name, in_ ,ksize, strides=[1,1,1,1], padding=g_.DEFAULT_PADDING, group=1, reuse=False):
    
    n_kern = ksize[3]
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides, padding=padding)

    with tf.variable_scope(name, reuse=reuse) as scope:
        if group == 1:
            kernel = _variable_with_weight_decay('weights', shape=ksize, wd=0.0)
            conv = convolve(in_, kernel)
        else:
            ksize[2] /= group
            kernel = _variable_with_weight_decay('weights', shape=ksize, wd=0.0)
        input_groups = tf.split(in_, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        # Concatenate the groups
        conv = tf.concat(output_groups, 3)

        biases = _variable_on_cpu('biases', [n_kern], tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(conv, name=scope.name)
        _activation_summary(conv)

    print(name, conv.get_shape().as_list())
    return conv

def _maxpool(name, in_, ksize, strides, padding=g_.DEFAULT_PADDING):
    pool = tf.nn.max_pool(in_, ksize=ksize, strides=strides,
                          padding=padding, name=name)

    print(name, pool.get_shape().as_list())
    return pool

def _fc(name, in_, outsize, dropout=1.0, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as scope:
        insize = in_.get_shape().as_list()[-1]
        weights = _variable_with_weight_decay('weights', shape=[insize, outsize], wd=0.004)
        biases = _variable_on_cpu('biases', [outsize], tf.constant_initializer(0.0))
        fc = tf.nn.relu(tf.matmul(in_, weights) + biases, name=scope.name)
        #fc = tf.nn.dropout(fc, dropout)

        _activation_summary(fc)

    print(name, fc.get_shape().as_list())
    return fc
    


def inference_multiview(views, n_classes, keep_prob):
    """
    views: N x V x W x H x C tensor
    """
    n_views = views.get_shape().as_list()[1] 

    # transpose views : (NxVxWxHxC) -> (VxNxWxHxC)
    views = tf.transpose(views, perm=[1, 0, 2, 3, 4])
    
    view_pool = []
    for i in range(n_views):
        # set reuse True for i > 0, for weight-sharing
        reuse = (i != 0)
        view = tf.gather(views, i) # NxWxHxC

        conv1 = _conv('conv1', view, [11, 11, 3, 96], [1, 4, 4, 1], 'VALID', reuse=reuse)
        lrn1 = None
        pool1 = _maxpool('pool1', conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv2 = _conv('conv2', pool1, [5, 5, 96, 256], group=2, reuse=reuse)
        lrn2 = None
        pool2 = _maxpool('pool2', conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        
        conv3 = _conv('conv3', pool2, [3, 3, 256, 384], reuse=reuse)
        conv4 = _conv('conv4', conv3, [3, 3, 384, 384], group=2, reuse=reuse)
        conv5 = _conv('conv5', conv4, [3, 3, 384, 256], group=2, reuse=reuse)

        pool5 = _maxpool('pool5', conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        
        view_pool.append(pool5)


    vp = _view_pool(view_pool,'vp')
    print('vp,',vp.get_shape().as_list())
    vp_new = tf.transpose(vp, perm=[1,2,3,0,4])
    print('vp_new,',vp_new.get_shape().as_list())
    dim = np.prod(vp_new.get_shape().as_list()[1:])
    reshape = tf.reshape(vp_new, [-1, dim])


    fc6 = _fc('fc6', reshape, 4096, dropout=keep_prob)
    fc7 = _fc('fc7', fc6, 4096, dropout=keep_prob)
    fc8 = _fc('fc8', fc7, n_classes)

    return fc8


def inference_multiview1(views, n_classes, keep_prob):
    """
    views: N x V x W x H x C tensor
    """
    n_views = views.get_shape().as_list()[1]

    # transpose views : (NxVxWxHxC) -> (VxNxWxHxC)
    views = tf.transpose(views, perm=[1, 0, 2, 3, 4])

    view_pool = []
    for i in range(n_views):
        # set reuse True for i > 0, for weight-sharing
        reuse = True
        view = tf.gather(views, i)  # NxWxHxC

        conv1 = _conv('conv1', view, [11, 11, 3, 96], [1, 4, 4, 1], 'VALID', reuse=reuse)
        lrn1 = None
        pool1 = _maxpool('pool1', conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv2 = _conv('conv2', pool1, [5, 5, 96, 256], group=2, reuse=reuse)
        lrn2 = None
        pool2 = _maxpool('pool2', conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv3 = _conv('conv3', pool2, [3, 3, 256, 384], reuse=reuse)
        conv4 = _conv('conv4', conv3, [3, 3, 384, 384], group=2, reuse=reuse)
        conv5 = _conv('conv5', conv4, [3, 3, 384, 256], group=2, reuse=reuse)

        pool5 = _maxpool('pool5', conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        view_pool.append(pool5)

    vp = _view_pool(view_pool,'vp')
    print('vp,',vp.get_shape().as_list())
    vp_new = tf.transpose(vp, perm=[1,2,3,0,4])
    print('vp_new,',vp_new.get_shape().as_list())
    dim = np.prod(vp_new.get_shape().as_list()[1:])
    reshape = tf.reshape(vp_new, [-1, dim])

    fc6 = _fc('fc6', reshape, 4096, dropout=keep_prob, reuse=True)
    fc7 = _fc('fc7', fc6, 4096, dropout=keep_prob, reuse=True)
    fc8 = _fc('fc8', fc7, n_classes, reuse=True)

    return fc8

def load_alexnet_to_mvcnn(sess, caffetf_modelpath):
    """ caffemodel: np.array, """

    caffemodel = np.load(caffetf_modelpath, encoding="latin1")
    data_dict = caffemodel.item()
    for l in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']:
        name = l
        _load_param(sess, name, data_dict[l])
    

def _load_param(sess, name, layer_data):
    w, b = layer_data

    with tf.variable_scope(name, reuse=True):
        for subkey, data in zip(('weights', 'biases'), (w, b)):
            print('loading ', name, subkey)

            try:
                var = tf.get_variable(subkey)
                sess.run(var.assign(data))
            except ValueError as e:
                print('varirable loading failed:', subkey, '(%s)' % str(e))


def _view_pool(view_features, name):
    vp = tf.expand_dims(view_features[0], 0) # eg. [100] -> [1, 100]
    for v in view_features[1:]:
        v = tf.expand_dims(v, 0)
        vp = tf.concat([vp, v], 0)
    return vp 


def loss(fc8, labels):
    l = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=fc8)
    l = tf.reduce_mean(l)
    
    tf.add_to_collection('losses', l)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def classify(fc8):
    softmax = tf.nn.softmax(fc8)
    y = tf.argmax(softmax, 1)
    return y


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
    total_loss: Total loss from loss().
    Returns:
    loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    print('losses:', losses)
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op

def loss_with_spring(o1, o2, y_):
    margin = 10.0
    labels_t = y_
    labels_f = tf.subtract(1.0, y_, name="1-yi")          # labels_ = !labels;
    eucd2 = tf.pow(tf.subtract(o1, o2), 2)
    eucd2 = tf.reduce_sum(eucd2, 1)
    eucd = tf.sqrt(eucd2+1e-6, name="eucd")
    C = tf.constant(margin, name="C")
    # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
    pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
    neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
    losses = tf.add(pos, neg, name="losses")
    loss = tf.reduce_mean(losses, name="loss")
    return loss

def train(total_loss, global_step, data_size):
    num_batches_per_epoch = data_size / g_.batch_size
    decay_steps = int(num_batches_per_epoch * g_.NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(g_.INIT_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    g_.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    train_op = tf.train.AdamOptimizer(lr).minimize(total_loss, global_step=global_step)

    return train_op
