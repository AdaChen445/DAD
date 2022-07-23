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
import os
import shutil
import argparse
import cv2
import random
from skimage.exposure import equalize_hist
from scipy.cluster.vq import whiten


ap = argparse.ArgumentParser()
ap.add_argument("-t", required=True)
ap.add_argument("-f", required=True)
ap.add_argument("-model")
args = vars(ap.parse_args())
label_type = str(args['t']) #ori/new
fea_type = str(args['f']) #specMfcc/melChroma
model_name = str(args['model']) #inseption/xception/dense121/dense201/resnet50/resnet50v2/vgg16

#########arguments##########
tsne_plot_name = model_name+'_'+label_type+'Label_'+fea_type
# resample_num = 3000
#########arguments##########


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def image_feature(img_path,model_name=None):
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
	elif model_name == 'ix':
		model0 = InceptionV3(weights='imagenet', include_top=False)
		input_preprocessing0 = pi_i
		model1 = Xception(weights='imagenet', include_top=False)
		input_preprocessing1 = pi_x
	elif model_name == 'id':
		model0 = InceptionV3(weights='imagenet', include_top=False)
		input_preprocessing0 = pi_i
		model1 = DenseNet121(weights='imagenet', include_top=False)
		input_preprocessing1 = pi_d
	elif model_name == 'xd':
		model0 = DenseNet121(weights='imagenet', include_top=False)
		input_preprocessing0 = pi_d
		model1 = Xception(weights='imagenet', include_top=False)
		input_preprocessing1 = pi_x
	elif model_name == 'ixd':
		model0 = InceptionV3(weights='imagenet', include_top=False)
		input_preprocessing0 = pi_i
		model1 = Xception(weights='imagenet', include_top=False)
		input_preprocessing1 = pi_x
		model2 = DenseNet121(weights='imagenet', include_top=False)
		input_preprocessing2 = pi_d


	input_size=224
	features = []
	img_name = []
	label = []
	audio_name = []
	# if len(img_path) > 30000: img_path = random.choices(img_path, k=resample_num) #resample
	for i in tqdm(img_path):
		fname=cluster_img_path+'/'+i
		img = cv2.imread(fname)
		img = cv2.resize(img, (input_size, input_size))
		#######CV preprocessing#######
		# img = equalize_hist(img)
		# img = whiten(img)
		#######CV preprocessing#######
		img = img.astype("float") / 255.0
		x = img_to_array(img)
		x=np.expand_dims(x,axis=0)

		if model_name in ('ix','id','xd'):
			x0=input_preprocessing0(x)
			feat0=model0.predict(x0)
			feat0=feat0.flatten()
			x1=input_preprocessing1(x)
			feat1=model1.predict(x1)
			feat1=feat1.flatten()
			feat = np.concatenate((feat0,feat1), axis=0)
		elif model_name == 'ixd':
			x0=input_preprocessing0(x)
			feat0=model0.predict(x0)
			feat0=feat0.flatten()
			x1=input_preprocessing1(x)
			feat1=model1.predict(x1)
			feat1=feat1.flatten()
			x2=input_preprocessing1(x)
			feat2=model2.predict(x2)
			feat2=feat2.flatten()
			feat = np.concatenate((feat0,feat1,feat2), axis=0)
		elif model_name == None:
			 feat=x.flatten()
		else:
			x=input_preprocessing(x)
			feat=model.predict(x)
			feat=feat.flatten()


		features.append(feat)
		img_name.append(i)
		label_list=i.split('_')
		# y=label_list[2]+'_'+label_list[3]+'_'+label_list[4]+'_'+label_list[5]
		y=label_list[1]+'_'+label_list[2]+'_'+label_list[3]+'_'+label_list[4] #for stage1
		y=y.split('.')[0]+'.wav'
		audio_name.append(y)

		if label_type == 'ori':
			label_name=label_list[2]+label_list[4]
			if label_name=='NGNG':
				label_num=1
			elif label_name=='NGOK':
				label_num=2
			else:
				label_num=3

		elif label_type == 'new':
			label_name=label_list[1]
			if label_name=='1':
				label_num=1
			elif label_name=='2':
				label_num=2
			elif label_name=='3':
				label_num=3
			elif label_name=='1loud':
				label_num=4
			elif label_name=='slap':
				label_num=5
			elif label_name=='loud':
				label_num=6
			elif label_name=='restart':
				label_num=7
			elif label_name=='ss':
				label_num=8
			elif label_name=='short':
				label_num=9
			elif label_name=='electro':
				label_num=10
			elif label_name=='pulse':
				label_num=11
			elif label_name in ('bad', 'badNG'):
				label_num=12
			elif label_name in ('good', 'goodN2O'):
				label_num=13
			else: label_num=14

		label.append(label_num)
	return features,img_name,label,audio_name

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
img_path=os.listdir(cluster_img_path)
img_features,img_name,label,audio_name=image_feature(img_path,model_name)
img_features_scatter = TSNE(n_components=2, init='random', perplexity=20).fit_transform(img_features)
# img_features_scatter = PCA(n_components=2).fit_transform(img_features)
img_features_scatter = scale_minmax(img_features_scatter)



print('[INFO] visualizing...')
font_size = 5
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4))
for i in range(img_features_scatter.shape[0]):
	ax1.text(img_features_scatter[i,0], img_features_scatter[i,1],str(label[i]), 
			color=color_list[label[i]], fontdict={'weight': 'bold', 'size': font_size})

	if label[i] in (1, 2, 3, 4, 11, 13):
		ax2.text(img_features_scatter[i,0], img_features_scatter[i,1],str(label[i]), 
			color='blue', fontdict={'weight': 'bold', 'size': font_size})
	else:
		ax2.text(img_features_scatter[i,0], img_features_scatter[i,1],str(label[i]), 
			color='red', fontdict={'weight': 'bold', 'size': font_size})
ax1.set_title('manual label 13 classes')
ax2.set_title('manual label NG / OK')
ax1.set_axis_off()
ax2.set_axis_off()
fig.savefig(tsne_plot_name)
