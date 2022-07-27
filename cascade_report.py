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
# IMG_HEIGHT = 200
IMG_WIDTH = 401
IMG_DEPTH = 3
BS = 32

# IMG_HEIGHT = 200
# dataset_folder = '../eval_SM_ok-ng'
# # dataset_folder = '../eval_SM_ok-n2o'
# model_dir_1 = './log/xceptl50_stage1_ok1-ng/best_0.9852.h5'
# model_dir_2 = './log/xceptl50_stage1_ok2-ng/best_0.9819.h5'
# model_dir_3 = './log/xceptl50_stage1_ok3-ng/best_0.9913.h5'
# model_dir_4 = './log/xceptl50_stage1_ok4-ng/best_0.9907.h5'

# IMG_HEIGHT = 140
# dataset_folder = '../eval_MC_ok-ng'
# # dataset_folder = '../eval_MC_ok-n2o'
# model_dir_1 = './log/xceptl50_stage1_MC_ok1-ng/best_0.9907.h5'
# model_dir_2 = './log/xceptl50_stage1_MC_ok2-ng/best_0.9941.h5'
# model_dir_3 = './log/xceptl50_stage1_MC_ok3-ng/best_0.9852.h5'
# model_dir_4 = './log/xceptl50_stage1_MC_ok4-ng/best_0.9931.h5'

IMG_HEIGHT = 168
dataset_folder = '../eval_MM_ok-ng'
# dataset_folder = '../eval_MM_ok-n2o'
model_dir_1 = './log/xceptl50_stage1_MM_ok1-ng/best_0.9857.h5'
model_dir_2 = './log/xceptl50_stage1_MM_ok2-ng/best_0.9836.h5'
model_dir_3 = './log/xceptl50_stage1_MM_ok3-ng/best_0.9854.h5'
model_dir_4 = './log/xceptl50_stage1_MM_ok4-ng/best_0.9943.h5'
model_dir_5 = './log/xceptl50_stage1_MM_ok5-ng/best_1.0000.h5'
#########arguments##########

from imutils import paths
imagePaths = list(paths.list_images(dataset_folder))
lebal_types = len(next(os.walk(dataset_folder))[1])
img = []
labels = []
img_ids = []
for imagePath in tqdm(imagePaths):
	label = imagePath.split(os.path.sep)[-2]
	img_id = imagePath.split(os.path.sep)[-1]
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
	image = preprocess_input(image)
	img.append(image)
	labels.append(label)
	img_ids.append(img_id)
img = np.array(img, dtype=np.float32)/255.0
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, lebal_types)
testX = img
testY = labels



def threshold_pred(prediction, threshold):
	thresholded_pred = []
	for i in range(len(prediction)):
		if prediction[i][1]>threshold: 
			thresholded_pred.append([0,1])
		else: thresholded_pred.append([1,0])
	return thresholded_pred

def model_predict(model_dir, testX, BS, le):
	model = load_model(model_dir)
	predictions = model.predict(x=testX, batch_size=BS)
	predictions = threshold_pred(predictions, 0.95)
	# print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_, digits=5))
	return predictions

pred_1 = model_predict(model_dir_1, testX, BS, le)
pred_2 = model_predict(model_dir_2, testX, BS, le)
pred_3 = model_predict(model_dir_3, testX, BS, le)
pred_4 = model_predict(model_dir_4, testX, BS, le)
pred_5 = model_predict(model_dir_5, testX, BS, le)



final_pred = []
for i in range(len(pred_1)):
	if np.argmax(pred_1[i]) or np.argmax(pred_2[i]) or np.argmax(pred_3[i]) or np.argmax(pred_4[i]) or np.argmax(pred_5[i]) == 1:
		final_pred.append([0,1])
	else: final_pred.append([1,0])
final_pred = np.array(final_pred, dtype=np.uint8)


### check id of mispredicts
for i in range(len(labels)):
	if labels[i][0]==1:
		if final_pred[i].argmax() != labels[i].argmax(): print(img_ids[i])


print(classification_report(testY.argmax(axis=1),
	final_pred.argmax(axis=1), target_names=le.classes_, digits=5))

ConfusionMatrixDisplay.from_predictions(testY.argmax(axis=1), 
	final_pred.argmax(axis=1), display_labels=le.classes_,colorbar=False, cmap='Blues')
plt.show()


