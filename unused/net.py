from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

kernal_h = 3
kernal_w = 3

class VGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(32, (kernal_h, kernal_w), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))	#32@64*64
		model.add(Conv2D(32, (kernal_h, kernal_w), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))	#32@64*64
		model.add(MaxPooling2D(pool_size=(2, 2)))	#32@32*32
		model.add(Dropout(0.25))

		model.add(Conv2D(64, (kernal_h, kernal_w), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))	#64@32*32
		model.add(Conv2D(64, (kernal_h, kernal_w), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))	#64@32*32
		model.add(MaxPooling2D(pool_size=(2, 2)))	#64@16*16
		model.add(Dropout(0.25))

		model.add(Conv2D(128, (kernal_h, kernal_w), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))	#128@16*16
		model.add(Conv2D(128, (kernal_h, kernal_w), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))	#128@16*16
		model.add(MaxPooling2D(pool_size=(2, 2)))	#128@8*8
		model.add(Dropout(0.25))

		# model.add(Conv2D(256, (kernal_h, kernal_w), padding="same"))
		# model.add(Activation("relu"))
		# model.add(BatchNormalization(axis=chanDim))	#128@16*16
		# model.add(Conv2D(256, (kernal_h, kernal_w), padding="same"))
		# model.add(Activation("relu"))
		# model.add(BatchNormalization(axis=chanDim))	#128@16*16
		# model.add(MaxPooling2D(pool_size=(2, 2)))	#128@8*8
		# model.add(Dropout(0.25))

		
		# FC => RELU layers
		model.add(Flatten())

		# model.add(Dense(64))
		# model.add(Activation("relu"))
		# model.add(BatchNormalization())
		# model.add(Dropout(0.5))

		model.add(Dense(32))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		model.add(Dense(16))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		model.add(Dense(8))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model