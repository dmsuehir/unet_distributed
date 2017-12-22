import os.path
import tensorflow as tf

def f1(y_true, y_pred):
	def recall(y_true, y_pred):
		"""Recall metric.

		Only computes a batch-wise average of recall.

		Computes the recall, a metric for multi-label classification of
		how many relevant items are selected.
		"""
		true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
		possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
		recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
		return recall

	def precision(y_true, y_pred):
		"""Precision metric.

		Only computes a batch-wise average of precision.

		Computes the precision, a metric for multi-label classification of
		how many selected items are relevant.
		"""
		true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
		predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
		precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
		return precision
	precision = precision(y_true, y_pred)
	recall = recall(y_true, y_pred)
	return ((precision*recall)/(precision+recall))


def dice_coef(y_true, y_pred, smooth = 1. ):
	y_true_f = tf.keras.backend.flatten(y_true)
	y_pred_f = tf.keras.backend.flatten(y_pred)
	intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
	coef = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
	return coef

def dice_coef_loss(y_true, y_pred):

	smooth = 1.
	y_true_f = tf.keras.backend.flatten(y_true)
	y_pred_f = tf.keras.backend.flatten(y_pred)
	intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
	loss = -tf.keras.backend.log(2. * intersection + smooth) + \
		tf.keras.backend.log((tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth))
	return loss

CHANNEL_LAST = True
if CHANNEL_LAST:
	concat_axis = -1
	data_format = 'channels_last'
	
else:
	concat_axis = 1
	data_format = 'channels_first'
	
tf.keras.backend.set_image_data_format(data_format)

def define_model(use_upsampling=False, 
	img_rows = 224, 
	img_cols = 224, 
	n_cl_in=3,
	n_cl_out=3, 
	dropout=0.2,
	print_summary = False):
	""" difference from model: img_rows and cols, order of axis, and concat_axis"""
	
	# Set keras learning phase to train
	tf.keras.backend.set_learning_phase(True)

	# Don't initialize variables on the fly
	tf.keras.backend.manual_variable_initialization(False)

	if CHANNEL_LAST:
		inputs = tf.keras.layers.Input((img_rows, img_cols, n_cl_in), name='Images')
	else:
		inputs = tf.keras.layers.Input((n_cl_in, img_rows, img_cols), name='Images')


	params = dict(kernel_size=(3, 3), activation='relu', 
				  padding='same', data_format=data_format,
				  kernel_initializer='he_uniform') #RandomUniform(minval=-0.01, maxval=0.01, seed=816))

	conv1 = tf.keras.layers.Conv2D(name='conv1a', filters=32, **params)(inputs)
	conv1 = tf.keras.layers.Conv2D(name='conv1b', filters=32, **params)(conv1)
	pool1 = tf.keras.layers.MaxPooling2D(name='pool1', pool_size=(2, 2))(conv1)

	conv2 = tf.keras.layers.Conv2D(name='conv2a', filters=64, **params)(pool1)
	conv2 = tf.keras.layers.Conv2D(name='conv2b', filters=64, **params)(conv2)
	pool2 = tf.keras.layers.MaxPooling2D(name='pool2', pool_size=(2, 2))(conv2)

	conv3 = tf.keras.layers.Conv2D(name='conv3a', filters=128, **params)(pool2)
	conv3 = tf.keras.layers.Dropout(dropout)(conv3) ### Trying dropout layers earlier on, as indicated in the paper
	conv3 = tf.keras.layers.Conv2D(name='conv3b', filters=128, **params)(conv3)
	
	pool3 = tf.keras.layers.MaxPooling2D(name='pool3', pool_size=(2, 2))(conv3)

	conv4 = tf.keras.layers.Conv2D(name='conv4a', filters=256, **params)(pool3)
	conv4 = tf.keras.layers.Dropout(dropout)(conv4) ### Trying dropout layers earlier on, as indicated in the paper
	conv4 = tf.keras.layers.Conv2D(name='conv4b', filters=256, **params)(conv4)
	
	pool4 = tf.keras.layers.MaxPooling2D(name='pool4', pool_size=(2, 2))(conv4)

	conv5 = tf.keras.layers.Conv2D(name='conv5a', filters=512, **params)(pool4)
	

	if use_upsampling:
		conv5 = tf.keras.layers.Conv2D(name='conv5b', filters=256, **params)(conv5)
		up6 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(name='up6', size=(2, 2))(conv5), conv4], axis=concat_axis)
	else:
		conv5 = tf.keras.layers.Conv2D(name='conv5b', filters=512, **params)(conv5)
		up6 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(name='transConv6', filters=256, data_format=data_format,
						   kernel_size=(2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=concat_axis)
		
	conv6 = tf.keras.layers.Conv2D(name='conv6a', filters=256, **params)(up6)
	

	if use_upsampling:
		conv6 = tf.keras.layers.Conv2D(name='conv6b', filters=128, **params)(conv6)
		up7 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(name='up7', size=(2, 2))(conv6), conv3], axis=concat_axis)
	else:
		conv6 = tf.keras.layers.Conv2D(name='conv6b', filters=256, **params)(conv6)
		up7 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(name='transConv7', filters=128, data_format=data_format,
						   kernel_size=(2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=concat_axis)

	conv7 = tf.keras.layers.Conv2D(name='conv7a', filters=128, **params)(up7)
	

	if use_upsampling:
		conv7 = tf.keras.layers.Conv2D(name='conv7b', filters=64, **params)(conv7)
		up8 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(name='up8', size=(2, 2))(conv7), conv2], axis=concat_axis)
	else:
		conv7 = tf.keras.layers.Conv2D(name='conv7b', filters=128, **params)(conv7)
		up8 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(name='transConv8', filters=64, data_format=data_format,
						   kernel_size=(2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=concat_axis)

	
	conv8 = tf.keras.layers.Conv2D(name='conv8a', filters=64, **params)(up8)
	
	if use_upsampling:
		conv8 = tf.keras.layers.Conv2D(name='conv8b', filters=32, **params)(conv8)
		up9 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(name='up9', size=(2, 2))(conv8), conv1], axis=concat_axis)
	else:
		conv8 = tf.keras.layers.Conv2D(name='conv8b', filters=64, **params)(conv8)
		up9 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(name='transConv9', filters=32, data_format=data_format,
						   kernel_size=(2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=concat_axis)


	conv9 = tf.keras.layers.Conv2D(name='conv9a', filters=32, **params)(up9)
	conv9 = tf.keras.layers.Conv2D(name='conv9b', filters=32, **params)(conv9)

	conv10 = tf.keras.layers.Conv2D(name='Mask', filters=n_cl_out, kernel_size=(1, 1), 
					data_format=data_format, activation='sigmoid')(conv9)

	model = tf.keras.models.Model(inputs=[inputs], outputs=[conv10])

	if print_summary:
		print (model.summary())	

	return model

