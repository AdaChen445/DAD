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


# model_dir = './log/xceptl50_stage1_ok-ngn2o/best_0.9404.h5'
model_dir = './log/xceptl50_n2ocleaner_ok-ng/best_0.9698.h5'
# model_dir = './log/xceptl50_n2ocleaner_okp-ng/best_0.9968.h5'
#########arguments##########


print("[INFO] loading images...")
from imutils import paths
imagePaths = list(paths.list_images(dataset_folder))
lebal_types = len(next(os.walk(dataset_folder))[1])
imgs = []
labels = []
img_ids = []
for imagePath in tqdm(imagePaths):
	label = imagePath.split(os.path.sep)[-2]
	img_id = imagePath.split(os.path.sep)[-1]
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
	image = preprocess_input(image)
	imgs.append(image)
	labels.append(label)
	img_ids.append(img_id)
imgs = np.array(imgs, dtype=np.float32)/255.0
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, lebal_types)
testX = imgs
testY = labels
# (trainX, testX, trainY, testY) = train_test_split(imgs, labels, test_size=0.25, random_state=42)


print("[INFO] evaluating...")
model = load_model(model_dir)
predictions = model.predict(x=testX, batch_size=BS)

### check id of mispredicts
# for i in range(len(imgs)):
# 	if labels[i][0]==1:
# 		if predictions[i].argmax() != labels[i].argmax(): print(img_ids[i])


# def threshold_pred(prediction, threshold):
# 	final_preds = []
# 	for i in range(len(prediction)):
# 		if prediction[i][1]>threshold:
# 			final_preds.append(1)
# 		else: final_preds.append(0)
# 	return final_preds
# final_preds = threshold_pred(predictions, 0.95)
final_preds = predictions.argmax(axis=1)


print(classification_report(testY.argmax(axis=1),
	final_preds, target_names=le.classes_, digits=5))

ConfusionMatrixDisplay.from_predictions(testY.argmax(axis=1), 
	final_preds, display_labels=le.classes_,colorbar=False, cmap='Blues')
plt.show()