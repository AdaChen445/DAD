from net import VGGNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from imutils import paths
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os
import datetime
import subprocess
from skimage.exposure import equalize_hist
from tqdm import tqdm

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet201
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.vgg16 import VGG16

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" #disable TF massages

#########arguments##########
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_DEPTH = 3
BS = 32
INIT_LR = 1e-3
#########arguments##########

ap = argparse.ArgumentParser()
ap.add_argument("-n", required=True) #name
ap.add_argument("-d", required=True) #datatype
ap.add_argument("-e", required=True) #epoch
args = vars(ap.parse_args())
model_name = str(args["n"])
data_type = str(args["d"])
EPOCHS = int(args['e'])
model_dir = './log/'+model_name
if not model_name == 'tt': os.mkdir(model_dir)
model_dir = model_dir+'/'
dataset_dir = '../dataset_'+data_type


print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset_dir))
lebal_types = len(next(os.walk(dataset_dir))[1])
data = []
labels = []
for imagePath in tqdm(imagePaths):
	label = imagePath.split(os.path.sep)[-2]
	# image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
	data.append(image)
	labels.append(label)
# data = np.array(data, dtype=np.float32) / 255.0 #scaling from 0 to 1
data = np.array(data, dtype=np.uint8) #to prevent ram EXPLOOOTION

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, lebal_types)

print(data.shape)
print(labels.shape)
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42)

##for singalchannel expending data depth
# trainX = tf.expand_dims(trainX, axis=-1)
# testX = tf.expand_dims(testX, axis=-1)

###data argumentation
# aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
# 	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
# 	horizontal_flip=True, fill_mode="nearest")


model_checkpoint_callback = ModelCheckpoint(
	filepath=model_dir+'best_{val_accuracy:.4f}.h5',
	save_weights_only=False,
	monitor='val_loss',
	mode='min',
	save_best_only=True)

print("[INFO] compiling model...")
model = VGGNet.build(width=IMG_WIDTH, height=IMG_HEIGHT, depth=IMG_DEPTH,
	classes=len(le.classes_))

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print("[INFO] training network for {} epochs...".format(EPOCHS))
H = model.fit(x=trainX, y=trainY, batch_size=BS,
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, callbacks = model_checkpoint_callback)

print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=BS)
f_report = str(classification_report(testY.argmax(axis=1),
		predictions.argmax(axis=1), target_names=le.classes_ ,digits=5))
print(f_report)
ConfusionMatrixDisplay.from_predictions(testY.argmax(axis=1), 
		predictions.argmax(axis=1), display_labels=le.classes_,colorbar=False, cmap='Blues')
plt.savefig(model_dir + "confusionMatrix.png")

# model.save(model_dir+"model.model", save_format="h5")
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