from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
import tensorflow as tf
import numpy as np
import pickle
import cv2
import os, glob
import matplotlib.pyplot as plt
import shutil



n2o_dir = '../n2o_SM'
model_name = 'xcep_tl100_n2ocleaner_db0.21'
model = load_model('./log/'+model_name+'/best_0.9936.h5')
le = pickle.loads(open('./log/'+model_name+'/le.pickle', 'rb').read())
imagePaths = glob.glob(os.path.join(n2o_dir, '*.png'))

result_dir = '../n2o_cleanResult/'
os.mkdir(result_dir)
os.mkdir(result_dir+'ok')
os.mkdir(result_dir+'n2o')

for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = np.array(image, dtype=np.float32) / 255.0
	image = np.expand_dims(image, axis=0)
	image = preprocess_input(image)
	preds = model.predict(image)[0]
	j = np.argmax(preds)
	label = le.classes_[j]

	if label == "ng":
		shutil.copy(imagePath, result_dir+'n2o/')
	else:
		shutil.copy(imagePath, result_dir+'ok/')
