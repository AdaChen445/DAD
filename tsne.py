from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet201
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.applications.inception_v3 import preprocess_input as pi_i
from tensorflow.keras.applications.xception import preprocess_input as pi_x
from tensorflow.keras.applications.densenet import preprocess_input as pi_d
from tensorflow.keras.applications.resnet import preprocess_input as pi_r
from tensorflow.keras.applications.resnet_v2 import preprocess_input as pi_r2
from tensorflow.keras.applications.vgg16 import preprocess_input as pi_v

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, DBSCAN, OPTICS
from sklearn.cluster import SpectralClustering, SpectralBiclustering, SpectralCoclustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize, suppress=True)
from tqdm import tqdm
import os, glob
import shutil
import argparse
import cv2
import random
from skimage.exposure import equalize_hist
from scipy.cluster.vq import whiten


ap = argparse.ArgumentParser()
ap.add_argument('-t', required=True)
ap.add_argument('-f', required=True)
ap.add_argument('-model', default=None)
args = vars(ap.parse_args())
machine_type = str(args['t'])
feature_type = str(args['f'])
model_name = str(args['model'])

#########arguments##########
img_path = '../dcase_t2/image/'+machine_type
tsne_plot_name = model_name+'_'+machine_type+'_'+feature_type
# resample_num = 3000
#########arguments##########


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def image_feature(img_path, model_name):
	if model_name == 'inception':
		model = InceptionV3(weights='imagenet', include_top=False)
		input_preprocessing = pi_i
	elif model_name == 'xception':
		model = Xception(weights='imagenet', include_top=False)
		input_preprocessing = pi_x
	elif model_name== 'dense121':
		model = DenseNet121(weights='imagenet', include_top=False)
		input_preprocessing = pi_d
	elif model_name== 'dense201':
		model = DenseNet201(weights='imagenet', include_top=False)
		input_preprocessing = pi_d
	elif model_name== 'resnet50':
		model = ResNet50(weights='imagenet', include_top=False)
		input_preprocessing = pi_r
	elif model_name== 'resnet50v2':
		model = ResNet50V2(weights='imagenet', include_top=False)
		input_preprocessing = pi_r2
	elif model_name== 'vgg16':
		model = VGG16(weights='imagenet', include_top=False)
		input_preprocessing = pi_v

	input_size=224
	img_features = []
	soundtype_id = []
	anomaly_status = []
	filenames = glob.glob(os.path.join(img_path, 'test/*.png')) + glob.glob(os.path.join(img_path, 'train/*.png'))
	# if len(filenames) > 30000: filenames = random.choices(filenames, k=resample_num) #resample

	for filename in tqdm(filenames):
		img = cv2.imread(filename)
		img = cv2.resize(img, (input_size, input_size))
		# img = equalize_hist(img)
		# img = whiten(img)
		img = img.astype("float") / 255.0
		x = img_to_array(img)
		x = np.expand_dims(x,axis=0)

		if model_name == 'None':
			feat=x.flatten()
		else:
			x=input_preprocessing(x)
			feat=model.predict(x)
			feat=feat.flatten()

		sound_id = int(filename.split(os.path.sep)[-1].split('_')[3])
		ano = filename.split(os.path.sep)[-1].split('_')[1]

		img_features.append(feat)
		soundtype_id.append(sound_id)
		anomaly_status.append(ano)

	return img_features,soundtype_id,anomaly_status

color_list = (
	'red',
	'limegreen',
	'royalblue',
	'purple',
	'olive',
	'dimgrey',
	'darkorange',
	'lawngreen',
	'darkorchid',
	'dodgerblue',
	'gold',
	'aqua',
	'crimson',
	'navy',
	'brown',
	'pink',
	'coral',
	'black',
	'lightgray',
	'lime',
	'b',
	'g'
	)


print('[INFO] feature extracting...')
img_features,soundtype_id,anomaly_status=image_feature(img_path,model_name)
img_features_scatter = TSNE(n_components=2, init='random', perplexity=20).fit_transform(img_features)
# img_features_scatter = PCA(n_components=2).fit_transform(img_features)
img_features_scatter = scale_minmax(img_features_scatter)



print('[INFO] visualizing...')
font_size = 5
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4))
for i in range(img_features_scatter.shape[0]):
	if anomaly_status[i]=='normal':
		ax1.text(img_features_scatter[i,0], img_features_scatter[i,1],str(soundtype_id[i]), 
				color=color_list[soundtype_id[i]], fontdict={'weight': 'bold', 'size': font_size})
	else:
		ax1.text(img_features_scatter[i,0], img_features_scatter[i,1],str(soundtype_id[i]), 
				color=color_list[soundtype_id[i]], alpha=0.2, fontdict={'weight': 'bold', 'size': font_size})
	if anomaly_status[i]=='normal':
		ax2.text(img_features_scatter[i,0], img_features_scatter[i,1],str(soundtype_id[i]),
			color='blue', fontdict={'weight': 'bold', 'size': font_size}, alpha=0.4)
	else:
		ax2.text(img_features_scatter[i,0], img_features_scatter[i,1],str(soundtype_id[i]),
			color='red', fontdict={'weight': 'bold', 'size': font_size}, alpha=0.4)
ax1.set_title('')
ax2.set_title('')
ax1.set_axis_off()
ax2.set_axis_off()
fig.savefig(tsne_plot_name)
