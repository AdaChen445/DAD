from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np
import cv2
import os, glob
from tqdm import tqdm
from tensorflow.keras.models import load_model

#########arguments##########
IMG_HEIGHT = 200
IMG_WIDTH = 401
IMG_DEPTH = 3
BS = 32
# dataset_folder = '../eval_ok-ngn2o'
# dataset_folder = '../eval_ok-ng'
dataset_folder = '../eval_ok-n2o'


# model_dir = './log/xceptl50_stage1_ok-ngn2o/best_0.9404.h5'
# model_dir = './log/xceptl50_stage1_ok21-ngn2o/best_0.9519.h5'
# model_dir = './log/xceptl50_stage1_ok23-ngn2o/best_0.9475.h5'
# model_dir = './log/xceptl50_stage1_ok25-ngn2o/best_0.9393.h5'

model_dir = './log/xceptl50_n2ocleaner_ok-ng/best_0.9698.h5'
# model_dir = './log/xceptl50_n2ocleaner_okp-ng/best_0.9968.h5'
# model_dir = './log/xceptl50_n2ocleaner_ok21-ng/best_0.9807.h5'
# model_dir = './log/xceptl50_n2ocleaner_ok23-ng/best_0.9817.h5'
# model_dir = './log/xceptl50_n2ocleaner_ok25-ng/best_0.9737.h5'
#########arguments##########


print("[INFO] loading images...")
from imutils import paths
imagePaths = list(paths.list_images(dataset_folder))
lebal_types = len(next(os.walk(dataset_folder))[1])
img = []
labels = []
for imagePath in tqdm(imagePaths):
	label = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
	image = preprocess_input(image)
	img.append(image)
	labels.append(label)
img = np.array(img, dtype=np.float32)/255.0
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, lebal_types)
# (trainX, testX, trainY, testY) = train_test_split(img, labels,
# 	test_size=0.25, random_state=42)
###already use eval-only dir
testX = img
testY = labels


print("[INFO] evaluating...")
model = load_model(model_dir)
predictions = model.predict(x=testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_, digits=5))

ConfusionMatrixDisplay.from_predictions(testY.argmax(axis=1), 
	predictions.argmax(axis=1), display_labels=le.classes_,colorbar=False, cmap='Blues')
plt.show()