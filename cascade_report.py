from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.xception import preprocess_input
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize, suppress=True)
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
dataset_folder = '../eval_ok-ng'
# dataset_folder = '../eval_ok-n2o'

model_dir_1 = './log/xceptl50_stage1_ok1-ng/best_0.9852.h5'
model_dir_2 = './log/xceptl50_stage1_ok2-ng/best_0.9778.h5'
model_dir_3 = './log/xceptl50_stage1_ok3-ng/best_0.9913.h5'
model_dir_4 = './log/xceptl50_stage1_ok4-ng/best_0.9907.h5'
#########arguments##########

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
testX = img
testY = labels


def model_predict(model_dir, testX, BS):
	model = load_model(model_dir)
	predictions = model.predict(x=testX, batch_size=BS)
	return predictions

pred_1 = model_predict(model_dir_1, testX, BS)
pred_2 = model_predict(model_dir_2, testX, BS)
pred_3 = model_predict(model_dir_3, testX, BS)
pred_4 = model_predict(model_dir_4, testX, BS)

final_pred = []
for i in range(len(pred_1)):
	if np.argmax(pred_1[i]) or np.argmax(pred_2[i]) or np.argmax(pred_3[i]) or np.argmax(pred_4[i]) == 1:
		final_pred.append([0,1])
	else: final_pred.append([1,0])
final_pred = np.array(final_pred, dtype=np.uint8)


print(classification_report(testY.argmax(axis=1),
	final_pred.argmax(axis=1), target_names=le.classes_, digits=5))

ConfusionMatrixDisplay.from_predictions(testY.argmax(axis=1), 
	final_pred.argmax(axis=1), display_labels=le.classes_,colorbar=False, cmap='Blues')
plt.show()