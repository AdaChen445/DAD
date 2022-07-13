from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os
from skimage.exposure import equalize_hist
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet201
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.resnet import ResNet50, ResNet152
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet152V2
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" #disable TF massages

#########arguments##########
# EPOCHS = 200
IMG_HIGHT = 140
IMG_WIDTH = 431
IMG_DEPTH = 3
BS = 8
INIT_LR = 1e-3
#########arguments##########

ap = argparse.ArgumentParser()
ap.add_argument("-n", required=True) #name
ap.add_argument("-m", required=True) #modle
ap.add_argument("-d", required=True) #datatype
ap.add_argument("-l", required=True) #loss func
ap.add_argument("-e", required=True) #epoch
args = vars(ap.parse_args())
model_name = str(args["n"])
model_type = str(args["m"]) 
data_type = str(args["d"])
loss_func = str(args['l'])
EPOCHS = int(args['e'])
model_dir = './log/'+model_name
if not model_name in ('tt', 'mir'): os.mkdir(model_dir)
model_dir = model_dir+'/'


print("[INFO] loading images...")
# data_folder = 'dataset_'+data_type
data_folder = 'mir_'+data_type ###MIR
imagePaths = list(paths.list_images('../'+data_folder))
lebal_types = len(next(os.walk('../'+data_folder))[1])
data = []
labels = []
for imagePath in tqdm(imagePaths):
	label = imagePath.split(os.path.sep)[-2]
	# image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
	image = cv2.imread(imagePath)

	### frequency masking for specmfcc
	# img1 = image[0:64, 0:401]
	# img2 = image[160:200, 0:401]
	# image = np.concatenate((img1,img2),axis=0)

	image = cv2.resize(image, (IMG_WIDTH, IMG_HIGHT))
	# img = equalize_hist(img) #only for grayscale
	data.append(image)
	labels.append(label)

data = np.array(data, dtype=np.float32) / 255.0 #scaling from 0 to 1
# data = np.array(data, dtype=np.uint8) #to prevent ram EXPLOOOTION

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, lebal_types)


if loss_func=='triplet':
	labels = labels[:,0]
	import tensorflow_addons as tfa
	loss_func = tfa.losses.TripletSemiHardLoss()


print(data.shape) #(bs,224,224,3)
print(labels.shape) #(bs,2)
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42)

##for singalchannel expending data depth
# trainX = tf.expand_dims(trainX, axis=-1)
# testX = tf.expand_dims(testX, axis=-1)


print("[INFO] compiling model...")
if model_type == 'res50':
	model = ResNet50(weights=None, include_top=True, input_shape=(IMG_HIGHT,IMG_WIDTH,IMG_DEPTH), classes=len(le.classes_))
elif model_type == 'res50v2':
	model = ResNet50V2(weights=None, include_top=True, input_shape=(IMG_HIGHT,IMG_WIDTH,IMG_DEPTH), classes=len(le.classes_))
elif model_type == 'res152':
	model = ResNet152(weights=None, include_top=True, input_shape=(IMG_HIGHT,IMG_WIDTH,IMG_DEPTH), classes=len(le.classes_))
elif model_type == 'res152v2':
	model = ResNet152V2(weights=None, include_top=True, input_shape=(IMG_HIGHT,IMG_WIDTH,IMG_DEPTH), classes=len(le.classes_))
elif model_type == 'incepres':
	model = InceptionResNetV2(weights=None, include_top=True, input_shape=(IMG_HIGHT,IMG_WIDTH,IMG_DEPTH), classes=len(le.classes_))
elif model_type == 'mobile':
	model = MobileNet(weights=None, include_top=True, input_shape=(IMG_HIGHT,IMG_WIDTH,IMG_DEPTH), classes=len(le.classes_))
elif model_type == 'dense121':
	model = DenseNet121(weights=None, include_top=True, input_shape=(IMG_HIGHT,IMG_WIDTH,IMG_DEPTH), classes=len(le.classes_))
elif model_type == 'dense201':
	model = DenseNet201(weights=None, include_top=True, input_shape=(IMG_HIGHT,IMG_WIDTH,IMG_DEPTH), classes=len(le.classes_))
elif model_type == 'inception':
	model = InceptionV3(weights=None, include_top=True, input_shape=(IMG_HIGHT,IMG_WIDTH,IMG_DEPTH), classes=len(le.classes_))
elif model_type == 'xception':
	model = Xception(weights=None, include_top=True, input_shape=(IMG_HIGHT,IMG_WIDTH,IMG_DEPTH), classes=len(le.classes_))
elif model_type == 'naslarge':
	model = NASNetLarge(weights=None, include_top=True, input_shape=(IMG_HIGHT,IMG_WIDTH,IMG_DEPTH), classes=len(le.classes_))
elif model_type == 'effib7':
	model = EfficientNetB7(weights=None, include_top=True, input_shape=(IMG_HIGHT,IMG_WIDTH,IMG_DEPTH), classes=len(le.classes_))
elif model_type == 'effiv2l':
	model = EfficientNetV2L(weights=None, include_top=True, input_shape=(IMG_HIGHT,IMG_WIDTH,IMG_DEPTH), classes=len(le.classes_))

model_checkpoint_callback = ModelCheckpoint(
	filepath=model_dir+'best_{val_accuracy:.4f}.h5',
	save_weights_only=False,
	monitor='val_accuracy',
	mode='max',
	save_best_only=True)


opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss=loss_func, optimizer=opt, metrics=["accuracy"])
H = model.fit(x=trainX, y=trainY, batch_size=BS,
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, callbacks = model_checkpoint_callback)


print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=BS)
f_report = str(classification_report(testY.argmax(axis=1),
		predictions.argmax(axis=1), target_names=le.classes_ ,digits=5))
print(f_report)
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

f = open(model_dir+"le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(model_dir + "plot.png")