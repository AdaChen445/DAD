from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from tensorflow.keras.utils import to_categorical
import numpy as np
import argparse
import cv2
import os, glob
from tqdm import tqdm
from tensorflow.keras.models import load_model

#########arguments##########
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_DEPTH = 3
BS = 32
#########arguments##########

ap = argparse.ArgumentParser()
ap.add_argument("-n", required=True) #name
ap.add_argument("-d", required=True) #datatype
ap.add_argument("-b", required=True) #bestmodelname
args = vars(ap.parse_args())
model_name = str(args["n"])
data_type = str(args["d"])
best_model = str(args["b"])
model_dir = './log/'+model_name+'/'

print("[INFO] loading images...")
dataset_folder = 'dataset_'+data_type
from imutils import paths
imagePaths = list(paths.list_images('../'+dataset_folder))
lebal_types = len(next(os.walk('../'+dataset_folder))[1])
img = []
labels = []
for imagePath in tqdm(imagePaths):
	label = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
	img.append(image)
	labels.append(label)
img = np.array(img, dtype=np.uint8) #to prevent ram EXPLOOOOOTION

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, lebal_types)

(trainX, testX, trainY, testY) = train_test_split(img, labels,
	test_size=0.25, random_state=42)

print("[INFO] loading model...")
model = load_model(model_dir+'best_'+str(best_model)+'.h5')

print("[INFO] evaluating...")
predictions = model.predict(x=testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_, digits=5))