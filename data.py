from preprocess import load_data, update_channels
import settings_dist
import numpy as np

def load_all_data():

	# Load train data
	print('-'*30)
	print('Loading and preprocessing train data...')
	print('-'*30)
	imgs_train, msks_train = load_data(settings_dist.OUT_PATH,"_train")
	imgs_train, msks_train = update_channels(imgs_train, msks_train, settings_dist.IN_CHANNEL_NO, settings_dist.OUT_CHANNEL_NO, settings_dist.MODE)

	# Load test data
	print('-'*30)
	print('Loading and preprocessing test data...')
	print('-'*30)
	imgs_test, msks_test = load_data(settings_dist.OUT_PATH,"_test")
	imgs_test, msks_test = update_channels(imgs_test, msks_test, settings_dist.IN_CHANNEL_NO, settings_dist.OUT_CHANNEL_NO, settings_dist.MODE)

	print("Training images shape: {}".format(imgs_train[0].shape))
	print("Training masks shape: {}".format(msks_train[0].shape))

	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)

	return imgs_train, msks_train, imgs_test, msks_test


def get_epoch(batch_size,imgs_train,msks_train):

	# Assuming imgs_train and msks_train are the same size
	train_size = imgs_train.shape[0]
	image_width = imgs_train.shape[1]
	image_height = imgs_train.shape[2]
	image_channels = imgs_train.shape[3]

	epoch_length = train_size - train_size%batch_size
	batch_count = epoch_length/batch_size

	# Shuffle and truncate arrays to equal 1 epoch
	zipped = zip(imgs_train,msks_train)
	np.random.shuffle(zipped)
	data,labels = zip(*zipped)
	data = np.asarray(data)[:epoch_length]
	labels = np.asarray(labels)[:epoch_length]

	# Reshape arrays into batch_count batches of length batch_size
	data = data.reshape((batch_count,batch_size,image_width,image_height,image_channels))
	labels = labels.reshape((batch_count,batch_size,image_width,image_height,image_channels))

	# Join batches of training examples with batches of labels
	epoch_of_batches = zip(data,labels)

	return epoch_of_batches


