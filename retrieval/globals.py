import numpy as np
import tensorflow as tf

"""
constants for the data set.
"""
DIM_DISCRIPTOR = 64

NUM_VIEWS = 20
TRAIN_LOL = ''
TEST_LOL = ''

"""
batch_size for testing
"""
batch_size = 100

"""
batch_size for training
"""
batch_size_genuine = 10
batch_size_impostor = 10

# this must be more than twice the BATCH_SIZE
INPUT_QUEUE_SIZE = 4 * batch_size

#Directory where to write event logs
train_dir = './tmp/'

#Whether to log device placement
log_device_placement = False

#Finetune with a pretrained model
weights = ''

#finetune with a model converted by caffe-tensorflow
#caffemodel = './alexnet_imagenet.npy'
#caffemodel = ''
caffemodel = './alexnet_imagenet.npy'
# caffemodel = '/data/alexnet_imagenet.npy'

# Constants describing the training process.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
WEIGHT_DECAY_FACTOR = 0.004 / 5. # 3500 -> 2.8
INIT_LEARNING_RATE = 0.0001

TOWER_NAME = 'tower'
DEFAULT_PADDING = 'SAME'

np.set_printoptions(precision=3)
