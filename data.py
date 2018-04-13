from preprocess import load_data, update_channels
import settings_dist
import numpy as np
import tensorflow as tf


def load_all_data():

    # Load train data
    tf.logging.info('-'*42)
    tf.logging.info('Loading and preprocessing training data...')
    tf.logging.info('-'*42)
    imgs_train, msks_train = load_data(settings_dist.OUT_PATH, "_train")

    # Load test data
    tf.logging.info('-'*38)
    tf.logging.info('Loading and preprocessing test data...')
    tf.logging.info('-'*38)
    imgs_test, msks_test = load_data(settings_dist.OUT_PATH, "_test")

    # # Dina: Temporarily use random data
    # imgs_train = np.random.rand(5000, 128, 128, 1)
    # msks_train = np.random.rand(5000, 128, 128, 1)
    # imgs_test = np.random.rand(1000, 128, 128, 1)
    # msks_test = np.random.rand(1000, 128, 128, 1)

    # Update channels
    imgs_train, msks_train = update_channels(imgs_train, msks_train,
                                             settings_dist.IN_CHANNEL_NO,
                                             settings_dist.OUT_CHANNEL_NO,
                                             settings_dist.MODE)
    imgs_test, msks_test = update_channels(imgs_test, msks_test,
                                           settings_dist.IN_CHANNEL_NO,
                                           settings_dist.OUT_CHANNEL_NO,
                                           settings_dist.MODE)

    tf.logging.info("Training images shape: {}".format(imgs_train.shape))
    tf.logging.info("Training masks shape:  {}".format(msks_train.shape))
    tf.logging.info("Testing images shape:  {}".format(imgs_test.shape))
    tf.logging.info("Testing masks shape:   {}".format(msks_test.shape))

    return imgs_train, msks_train, imgs_test, msks_test


def get_epoch(batch_size, imgs_train, msks_train):

    # Assuming imgs_train and msks_train are the same size
    train_size = imgs_train.shape[0]
    image_width = imgs_train.shape[1]
    image_height = imgs_train.shape[2]
    image_channels = imgs_train.shape[3]

    epoch_length = train_size - train_size % batch_size
    batch_count = int(epoch_length/batch_size)

    # Shuffle and truncate arrays to equal 1 epoch
    zipped = list(zip(imgs_train, msks_train))
    np.random.shuffle(zipped)
    data, labels = zip(*zipped)
    data = np.asarray(data)[:epoch_length]
    labels = np.asarray(labels)[:epoch_length]

    # Reshape arrays into batch_count batches of length batch_size
    data = data.reshape((batch_count, batch_size, image_width, image_height,
                         image_channels))
    labels = labels.reshape((batch_count, batch_size, image_width,
                             image_height, image_channels))

    # Join batches of training examples with batches of labels
    epoch_of_batches = list(zip(data, labels))

    return np.array(epoch_of_batches)
