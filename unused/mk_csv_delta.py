from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from imutils import paths
import pickle
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import random
import subprocess
import argparse
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print('[INFO] executing: mk_csv_delta.py')
ap = argparse.ArgumentParser()
ap.add_argument("-n", required=True)
args = vars(ap.parse_args())
model_name = str(args["n"])
model_dir='./log/'+model_name+'/'

data_width = 128
data_height = 128

model = load_model(model_dir+'best_0.9986.h5')
le = pickle.loads(open(model_dir+"le.pickle", "rb").read())

anomaly_score = []
anomaly = []
train = []
ng2ok = []

def inverte(x):
	return abs(1-x)

# #train_noramal
# print('[INFO] now converting: train_noramal')
# imagePaths = list(paths.list_images("./dataset/ok_train")) #1000
# imagePaths = random.choices(imagePaths, k=1000)
# for imagePath in imagePaths:
# 	image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
# 	image = cv2.resize(image, (data_width, data_height))
# 	image = image.astype("float") / 255.0
# 	image = img_to_array(image)
# 	image = np.expand_dims(image, axis=0)
# 	pred = model.predict(image)[0][0]

# 	# preds = model.predict(image)[0]
# 	# j = np.argmax(preds)
# 	# print(preds)
# 	# label = le.classes_[j]
# 	# print(label)

# 	anomaly_score.append(inverte(pred))
# 	anomaly.append(False)
# 	train.append(True)
# 	ng2ok.append(False)

# #train_outlier
# print('[INFO] now converting: train_outlier')
# imagePaths = list(paths.list_images("./dataset/outlier")) #1000
# imagePaths = random.choices(imagePaths, k=1000)
# for imagePath in imagePaths:
# 	image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
# 	image = cv2.resize(image, (data_width, data_height))
# 	image = image.astype("float") / 255.0
# 	image = img_to_array(image)
# 	image = np.expand_dims(image, axis=0)
# 	pred = model.predict(image)[0][0]

# 	anomaly_score.append(inverte(pred))
# 	anomaly.append(True)
# 	train.append(True)
# 	ng2ok.append(False)

#test_ok
print('[INFO] now converting: test_ok')
imagePaths = list(paths.list_images("../stage1_eval/ok")) #1000
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	if image.shape != (200,401,3):
		print(imagePath)
		print(image.shape)
		continue
	image = np.array(image, dtype=np.float32)/255.0
	image = np.expand_dims(image, axis=0)
	pred = model.predict(image)[0][1]

	anomaly_score.append(inverte(pred))
	anomaly.append(False)
	train.append(False)
	ng2ok.append(False)

#test_ng
print('[INFO] now converting: test_ng')
imagePaths = list(paths.list_images("../stage1_eval/ng")) #2271
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	if image.shape != (200,401,3):
		print(imagePath)
		print(image.shape)
		continue
	image = np.array(image, dtype=np.float32)/255.0
	image = np.expand_dims(image, axis=0)
	pred = model.predict(image)[0][1]

	anomaly_score.append(inverte(pred))
	anomaly.append(True)
	train.append(False)
	ng2ok.append(False)

#test_ng2ok
print('[INFO] now converting: test_ng2ok')
imagePaths = list(paths.list_images("../stage1_eval/ng2ok")) #1544
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	if image.shape != (200,401,3):
		print(imagePath)
		print(image.shape)
		continue
	image = np.array(image, dtype=np.float32)/255.0
	image = np.expand_dims(image, axis=0)
	pred = model.predict(image)[0][1]

	anomaly_score.append(inverte(pred))
	anomaly.append(True)
	train.append(False)
	ng2ok.append(True)


dict = {'anomaly':anomaly, 'train':train, 'ng2ok':ng2ok, 'anomaly_score':anomaly_score}
df = pd.DataFrame(dict)
df.to_csv(model_dir+model_name+'.csv')

# subprocess.run("python mk_plot_delta.py -n "+model_name,shell=True)