from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2
import os, glob
from tqdm import tqdm
from tensorflow.keras.models import load_model

#########arguments##########
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_DEPTH = 3
BS = 32
dataset_folder = 'dataset_stage1'
model_dir = './log/xcepttl50_stage1/best_0.9404.h5'
#########arguments##########


print("[INFO] loading images...")
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
model = load_model(model_dir)

print("[INFO] evaluating...")
predictions = model.predict(x=testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_, digits=5))