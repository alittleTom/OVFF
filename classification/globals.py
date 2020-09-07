"""
constants for the data set.
ModelNet40 for example
"""
NUM_CLASSES = 20

NUM_VIEWS = 20
TRAIN_LOL = './train_lists.txt'
TEST_LOL = './test_lists.txt'


"""
constants for both training and testing
"""
BATCH_SIZE = 32

# this must be more than twice the BATCH_SIZE
INPUT_QUEUE_SIZE = 4 * BATCH_SIZE


"""
constants for training the model
"""
INIT_LEARNING_RATE = 0.0001


# sample how many shapes for validation
# this affects the validation time
VAL_SAMPLE_SIZE = 660

# do a validation every VAL_PERIOD iterations
VAL_PERIOD = 30

# save the progress to checkpoint file every SAVE_PERIOD iterations
# this takes tens of seconds. Don't set it smaller than 100.
SAVE_PERIOD = 50

