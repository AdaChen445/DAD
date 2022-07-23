from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os, glob
import PIL
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage.exposure import equalize_hist
import sys
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam, Nadam, Ftrl, SGD, Adagrad, Adamax, Adadelta
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from vit_keras import  vit
import random
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet201
from tensorflow.keras.applications.resnet import ResNet50, ResNet152
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet152V2
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L
from tensorflow.keras.applications.inception_v3 import preprocess_input as pi_i
from tensorflow.keras.applications.xception import preprocess_input as pi_x
from tensorflow.keras.applications.densenet import preprocess_input as pi_d
from tensorflow.keras.applications.resnet import preprocess_input as pi_r
from tensorflow.keras.applications.resnet_v2 import preprocess_input as pi_r2
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as pi_e
import tensorflow as tf


#########arguments##########
IMG_HIGHT = 200
IMG_WIDTH = 401
IMGSIZE_VIT = 224
IMG_DEPTH = 3
BS = 8
INIT_LR = 1e-4
FREEZE_PERCENT = 0
is3channel = False
#########arguments##########


ap = argparse.ArgumentParser()
ap.add_argument("-n", required=True) #name
ap.add_argument("-m", required=True) #modle
ap.add_argument("-d", required=True) #datatype
ap.add_argument("-l", required=True) #loss func
ap.add_argument("-e", required=True) #epoch tl
ap.add_argument("-c") #3channel 
ap.add_argument("-tt") #test object
args = vars(ap.parse_args())
model_name = str(args["n"])
model_type = str(args["m"])
data_type = str(args["d"])
loss_func = str(args['l'])
epoch = int(args['e'])
is3channel = bool(args['c'])
acti = str(args['tt'])

model_dir = './log/'+model_name
if not model_name in ('tt', 'vit'): os.mkdir(model_dir)
model_dir = model_dir+'/'
if model_type=='vit':
	IMG_WIDTH = IMGSIZE_VIT
	IMG_HIGHT = IMGSIZE_VIT



print("[INFO] loading data...")
data_folder = '../dataset_'+data_type
if is3channel:
	dataset_folder = data_folder+'/c1'
else:
	dataset_folder = data_folder
from imutils import paths
imagePaths = list(paths.list_images(dataset_folder))
lebal_types = len(next(os.walk(dataset_folder))[1])
img = []
labels = []
for imagePath in tqdm(imagePaths):
	label = imagePath.split(os.path.sep)[-2]
	### image resizing
	# image = Image.open(imagePath).convert('RGB')
	# if resize_method==1:
	# 	image = image.resize((IMG_HIGHT, IMG_WIDTH), resample=PIL.Image.BICUBIC)
	# elif resize_method==2:
	# 	image = image.resize((IMG_HIGHT, IMG_WIDTH), resample=PIL.Image.BILINEAR)
	# elif resize_method==3:
	# 	image = image.resize((IMG_HIGHT, IMG_WIDTH), resample=PIL.Image.HAMMING)
	# elif resize_method==4:
	# 	image = image.resize((IMG_HIGHT, IMG_WIDTH), resample=PIL.Image.LANCZOS)
	# elif resize_method==5:
	# 	image = image.resize((IMG_HIGHT, IMG_WIDTH), resample=PIL.Image.BOX)
	# elif resize_method==6:
	# 	image = image.resize((IMG_HIGHT, IMG_WIDTH), resample=PIL.Image.NEAREST)
	# image = cv2.imread(imagePath) #cv2 default is inter_linear
	# if resize_method==7:
	# 	image = cv2.resize(image,(IMG_HIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)
	# elif resize_method==8:
	# 	image = cv2.resize(image,(IMG_HIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
	# elif resize_method==9:
	# 	image = cv2.resize(image,(IMG_HIGHT, IMG_WIDTH), interpolation=cv2.INTER_NEAREST)

	if is3channel:
		###custom 3-channel
		imagePath1 = imagePath
		imagePath2 = '../'+data_folder+'/c2/'+label+'/'+imagePath.split(os.path.sep)[-1]
		imagePath3 = '../'+data_folder+'/c3/'+label+'/'+imagePath.split(os.path.sep)[-1]
		img1 = cv2.imread(imagePath1, cv2.IMREAD_GRAYSCALE)
		img1 = cv2.resize(img1,(IMG_WIDTH, IMG_HIGHT))#, interpolation=cv2.INTER_CUBIC)
		img1 = np.array(img1, dtype=np.uint8)
		img2 = cv2.imread(imagePath2, cv2.IMREAD_GRAYSCALE)
		img2 = cv2.resize(img2,(IMG_WIDTH, IMG_HIGHT))#, interpolation=cv2.INTER_CUBIC)
		img2 = np.array(img2, dtype=np.uint8)
		img3 = cv2.imread(imagePath3, cv2.IMREAD_GRAYSCALE)
		img3 = cv2.resize(img3,(IMG_WIDTH, IMG_HIGHT))#, interpolation=cv2.INTER_CUBIC)
		img3 = np.array(img3, dtype=np.uint8)
		image = np.dstack([img1, img2, img3]).astype(np.uint8)
	else:
		image = cv2.imread(imagePath)
		if image.shape != (IMG_HIGHT,IMG_WIDTH,3):
			print(imagePath, image.shape)
			continue
		image = cv2.resize(image, (IMG_WIDTH, IMG_HIGHT))
		image = np.array(image, dtype=np.uint8)
	# image = equalize_hist(image) #this return float
	img.append(image)
	labels.append(label)

img = np.array(img, dtype=np.float32)/255.0
le = LabelEncoder()
labels = le.fit_transform(labels)
f = open(model_dir+"le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()

if loss_func=='triplet':
	import tensorflow_addons as tfa
	loss_func = tfa.losses.TripletSemiHardLoss()
else:
	labels = to_categorical(labels, lebal_types)

print(np.array(img).shape, np.array(labels).shape)
(trainX, testX, trainY, testY) = train_test_split(img, labels, test_size=0.25, random_state=42)



print("[INFO] compiling model...")
if model_type == 'res50':
	base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HIGHT,IMG_WIDTH,IMG_DEPTH))
	input_preprocessing = pi_r
	freeze_layer = round(107*FREEZE_PERCENT)
elif model_type == 'res50v2':
	base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(IMG_HIGHT,IMG_WIDTH,IMG_DEPTH))
	input_preprocessing = pi_r2
	freeze_layer = round(103*FREEZE_PERCENT)
elif model_type == 'res152':
	base_model = ResNet152(weights='imagenet', include_top=False, input_shape=(IMG_HIGHT,IMG_WIDTH,IMG_DEPTH))
	input_preprocessing = pi_r
	freeze_layer = round(311*FREEZE_PERCENT)
elif model_type == 'res152v2':
	base_model = ResNet152V2(weights='imagenet', include_top=False, input_shape=(IMG_HIGHT,IMG_WIDTH,IMG_DEPTH))
	input_preprocessing = pi_r2
	freeze_layer = round(307*FREEZE_PERCENT)
elif model_type == 'dense121':
	base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_HIGHT,IMG_WIDTH,IMG_DEPTH))
	input_preprocessing = pi_d
	freeze_layer = round(242*FREEZE_PERCENT)
elif model_type == 'dense201':
	base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(IMG_HIGHT,IMG_WIDTH,IMG_DEPTH))
	input_preprocessing = pi_d
	freeze_layer = round(402*FREEZE_PERCENT)
elif model_type == 'inception':
	base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_HIGHT,IMG_WIDTH,IMG_DEPTH))
	input_preprocessing = pi_i
	freeze_layer = round(189*FREEZE_PERCENT)
elif model_type == 'xception':
	base_model = Xception(weights='imagenet', include_top=False)#, input_shape=(IMG_HIGHT,IMG_WIDTH,IMG_DEPTH))
	input_preprocessing = pi_x
	freeze_layer = round(81*FREEZE_PERCENT)
elif model_type == 'effiv2l':
	base_model = EfficientNetV2L(weights='imagenet', include_top=False, input_shape=(IMG_HIGHT,IMG_WIDTH,IMG_DEPTH))
	input_preprocessing = pi_e
	freeze_layer = round(956*FREEZE_PERCENT)
elif model_type == 'vit':
	base_model = vit.vit_b16(image_size=IMGSIZE_VIT, pretrained=True, include_top=False, pretrained_top=False)
elif model_type == 'stack':
	base_model_1 = Xception(weights='imagenet', include_top=False)
	base_model_2 = vit.vit_b16(image_size=64, pretrained=True, include_top=False, pretrained_top=False)
	input_preprocessing = pi_x
else: raise


if model_type=='vit':
	base_model.trainable = True
	prediction_layer = tf.keras.layers.Dense(lebal_types, activation='sigmoid')
	inputs = tf.keras.Input(shape=(IMGSIZE_VIT, IMGSIZE_VIT, 3))
	x = base_model(inputs, training=True)
	x = tf.keras.layers.Dropout(0.1)(x)
	outputs = prediction_layer(x)
	model = tf.keras.Model(inputs, outputs)
	model.summary()
elif model_type=='stack':
	trainX = input_preprocessing(trainX)
	testX = input_preprocessing(testX)
	base_model_1.trainable = True
	base_model_2.trainable = True
	inputs = tf.keras.Input(shape=(300, 300, 3))
	x = input_preprocessing(inputs)
	x = base_model_1(x, training=True)
	print(x.shape)
	x =  tf.keras.layers.GlobalAveragePooling2D()(x)
	print(x.shape)
	x = tf.keras.layers.RepeatVector(3)(x)
	print(x.shape)
	x = tf.keras.layers.Permute((2,1))(x)
	print(x.shape)
	x = tf.keras.layers.UpSampling1D(2)(x)
	print(x.shape)
	x = tf.keras.layers.Reshape((64,64,3))(x)
	print(x.shape)
	x = base_model_2(x, training=True)
	x = tf.keras.layers.Dropout(0.1)(x)
	outputs = tf.keras.layers.Dense(lebal_types, activation='sigmoid')(x)
	model = tf.keras.Model(inputs, outputs)
	model.summary()
else:
	### normal cnn model
	trainX = input_preprocessing(trainX)
	testX = input_preprocessing(testX)
	base_model.trainable = True
	if FREEZE_PERCENT>0:
		for layer in base_model.layers[:freeze_layer]:
			layer.trainable = False
	inputs = tf.keras.Input(shape=(IMG_HIGHT, IMG_WIDTH, IMG_DEPTH))
	x = input_preprocessing(inputs)
	x = base_model(x, training=True)
	x = tf.keras.layers.GlobalAveragePooling2D()(x)
	x = tf.keras.layers.Dropout(0.2)(x)
	outputs = tf.keras.layers.Dense(lebal_types, activation='sigmoid')(x)
	model = tf.keras.Model(inputs, outputs)
	model.summary()


model_checkpoint_callback = ModelCheckpoint(
	filepath=model_dir+'best_{val_accuracy:.4f}.h5',
	save_weights_only=False,
	monitor='val_loss',
	mode='min',
	save_best_only=True)

opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / epoch)
# opt = Nadam(learning_rate=INIT_LR, beta_1=0.9, beta_2=0.9)
# opt = Adamax(learning_rate=INIT_LR, decay=INIT_LR / epoch)
# opt = SGD(learning_rate=INIT_LR, decay=INIT_LR / epoch)

model.compile(loss=loss_func, optimizer=opt, metrics=["accuracy"])
H = model.fit(x=trainX, y=trainY, batch_size=BS,
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=epoch, callbacks = model_checkpoint_callback)



print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=BS)
f_report = str(classification_report(testY.argmax(axis=1),
		predictions.argmax(axis=1), target_names=le.classes_ ,digits=5))
print(f_report)
ConfusionMatrixDisplay.from_predictions(testY.argmax(axis=1), 
		predictions.argmax(axis=1), display_labels=le.classes_,colorbar=False, cmap='Blues')
plt.savefig(model_dir + "confusionMatrix.png")

f = open(model_dir+"classification_report.txt", "w")
f.write(f_report)
f.write('acc')
f.write(str(H.history["accuracy"])+'\n')
f.write('val_acc')
f.write(str(H.history["val_accuracy"])+'\n')
f.write('val_loss')
f.write(str(H.history["val_loss"])+'\n')
f.write('loss')
f.write(str(H.history["loss"])+'\n')
f.close()

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epoch), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epoch), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epoch), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epoch), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(model_dir + "plot_tl.png")