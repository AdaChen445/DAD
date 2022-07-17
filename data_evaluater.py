from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import cv2
import os, glob
import matplotlib.pyplot as plt
import shutil

IMG_HIGHT = 200
IMG_WIDTH = 401
val_ng_dir = '../ng_SM/val'
val_n2o_dir = '../ng2ok_SM/val'
val_ok_dir = '../ok_SM/val'
# val_model_name = 'xceptl50_n2ocleaner_okp-ng'
# model = load_model('./log/'+val_model_name+'/best_0.9968.h5')
val_model_name = 'xceptl50_stage1_okc-ngn2o'
model = load_model('./log/'+val_model_name+'/best_0.9519.h5')
le = pickle.loads(open('./log/'+val_model_name+'/le.pickle', 'rb').read())

###0717 have some serious issue, switch to classfy_report.py to eval model

pred_label = []
true_label = []
label_name = ['ng+n2o', 'ok']
pred_label_n2ocleaner = []
true_label_n2ocleaner = []
label_name_n2ocleaner = ['ng', 'ok']


# print('[INFO]evaluating: ng')
# imagePaths = glob.glob(os.path.join(val_ng_dir, '*.png'))
# ng_ng=0 
# ng_ok=0
# for imagePath in imagePaths:
# 	image = cv2.imread(imagePath)
# 	image = np.array(image, dtype=np.float32) / 255.0
# 	image = np.expand_dims(image, axis=0)
# 	image = preprocess_input(image)
# 	preds = model.predict(image)[0]
# 	j = np.argmax(preds)
# 	# label = le.classes_[j]

# 	pred_label.append(j)
# 	true_label.append(0)
# 	pred_label_n2ocleaner.append(j)
# 	true_label_n2ocleaner.append(0)
# 	if j==0: 
# 		ng_ng+=1
# 	else: 
# 		ng_ok+=1
# print(ng_ok, ng_ng)


# print('[INFO]evaluating: n2o')
# imagePaths = glob.glob(os.path.join(val_n2o_dir, '*.png'))
# n2o_ng=0 
# n2o_ok=0
# for imagePath in imagePaths:
# 	image = cv2.imread(imagePath)
# 	image = np.array(image, dtype=np.float32) / 255.0
# 	image = np.expand_dims(image, axis=0)
# 	image = preprocess_input(image)
# 	preds = model.predict(image)[0]
# 	j = np.argmax(preds)
# 	# label = le.classes_[j]

# 	pred_label.append(j)
# 	true_label.append(0)
# 	if j==0: 
# 		n2o_ng+=1
# 	else: 
# 		n2o_ok+=1
# print(n2o_ok, n2o_ng)



print('[INFO]evaluating: ok')
imagePaths = glob.glob(os.path.join(val_ok_dir, '*.png'))
ok_ng=0 
ok_ok=0
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	if image.shape != (IMG_HIGHT,IMG_WIDTH,3): continue
	###
	# image = cv2.resize(image, (IMG_WIDTH, IMG_HIGHT))
	# image = np.array(image, dtype=np.uint8)
	###
	image = np.array(image, dtype=np.float32) / 255.0
	image = np.expand_dims(image, axis=0)
	image = preprocess_input(image)
	preds = model.predict(image)[0]
	j = np.argmax(preds)
	label = le.classes_[j]
	print(j, label)

	pred_label.append(j)
	true_label.append(1)
	pred_label_n2ocleaner.append(j)
	true_label_n2ocleaner.append(1)
	if j==0: 
		ok_ng+=1
	else: 
		ok_ok+=1
print(ok_ok, ok_ng)


### ok/ng
print(classification_report(true_label_n2ocleaner, pred_label_n2ocleaner, target_names=label_name_n2ocleaner ,digits=5))

### ok/ngn2o
print(classification_report(true_label, pred_label, target_names=label_name ,digits=5))