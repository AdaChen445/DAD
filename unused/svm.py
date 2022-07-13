from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from imutils import paths
import numpy as np
import cv2
import os
from sklearn.svm import SVC

###use model
# from skimage.exposure import equalize_hist
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.applications.densenet import DenseNet121
# from tensorflow.keras.applications.densenet import preprocess_input
# model = DenseNet121(weights='imagenet', include_top=False)

#########arguments##########
IMG_HEIGHT = 224
IMG_WIDTH = 224
#########arguments##########

# print('[INFO]extracting feature')
imagePaths = list(paths.list_images("../dataset"))
lebal_types = len(next(os.walk('../dataset'))[1])

accu_record = 0.78

for upper_bond in range(500, 946, 3):
	for under_bond in range(upper_bond+20, 946,3):
		print((upper_bond, under_bond))
		data = []
		labels = []
		for imagePath in imagePaths:
			label = imagePath.split(os.path.sep)[-2]

			###use SVM only
			image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
			# feat = image

			###crop and concate
			img1 = image[0:upper_bond, 0:802]
			img2 = image[under_bond:946, 0:802]
			feat = np.concatenate((img1,img2),axis=0)

			###use model as extracter
			# image = cv2.imread(imagePath)
			# image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
			# image = equalize_hist(image)
			# image = image.astype("float") / 255.0
			# x = img_to_array(image)
			# x=np.expand_dims(x,axis=0)
			# x=preprocess_input(x)
			# feat=model.predict(x)
			# feat=feat.flatten()

			data.append(feat)
			labels.append(label)

		###use SVM only
		data = np.array(data, dtype=np.uint8).reshape(len(data),-1)

		(trainX, testX, trainY, testY) = train_test_split(data, labels,
			test_size=0.25, random_state=42)

		# print('[INFO]training')
		model = SVC(kernel='linear',gamma='auto')
		H = model.fit(trainX, trainY)
		predictions = model.predict(testX)
		accu = classification_report(testY,predictions ,digits=5, output_dict=True)['accuracy']
		if accu > accu_record:
			accu_record=accu
			print(accu)
			f=open('svm_report.txt','a')
			f.write('ub: '+str(upper_bond))
			f.write('  lb: '+str(under_bond)+'\n')
			f.write(str(classification_report(testY,predictions ,digits=5)))
			f.write('=========='+'\n')
			f.close()
