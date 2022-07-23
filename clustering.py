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
ap.add_argument("-f", required=True)
ap.add_argument("-c", required=False)
ap.add_argument("-model", required=True)
ap.add_argument('-eps')
args = vars(ap.parse_args())
fea_type = str(args['f']) #spectrogram/specMfcc/specChroma/specMfccChroma/melSpec/melChroma or others
model_name = str(args['model']) #inseption/xception/dense121/dense201/resnet50/resnet50v2/vgg16
cluster_type = str(args['c'])  #km/ap/ac/db/op/sp/sb/sc
# eps = float(args['eps'])

#########arguments##########
cluster_img_path = '../ok_SM/test_train'
cluster_audio_dir = '../ok_audio'

cluster_type_dir = '../'+cluster_type
cluster_result_dir = cluster_type_dir+'/'+model_name+'_'+fea_type+'_clusterResult'
# cluster_result_dir = cluster_type_dir+'/'+str(eps)

cluster_plot_name = cluster_result_dir+'/'+model_name+'_'+fea_type+'_'+cluster_type

cluster_num = 20
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

	input_size=224
	features = []
	img_name = []
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

		if model_name == None:
			 feat=x.flatten()
		else:
			x=input_preprocessing(x)
			feat=model.predict(x)
			feat=feat.flatten()

		features.append(feat)
		img_name.append(i)
		label_list=i.split('_')
		y=label_list[1]+'_'+label_list[2]+'_'+label_list[3]+'_'+label_list[4] #for stage1
		audio_name.append(y.replace('.png', '.wav'))

	return features,img_name,audio_name

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
img_features,img_name,audio_name=image_feature(img_path,model_name)
img_features_scatter = TSNE(n_components=2, init='random', perplexity=20).fit_transform(img_features)
# img_features_scatter = PCA(n_components=2).fit_transform(img_features)
img_features_scatter = scale_minmax(img_features_scatter)


print('[INFO] clustering...')
classes = cluster_num
if cluster_type == 'km':
	clusters = KMeans(n_clusters=classes, random_state=42, init='k-means++')
elif cluster_type == 'ac':
	clusters = AgglomerativeClustering(n_clusters=classes, linkage='ward')
elif cluster_type == 'gm':
	clusters = GaussianMixture(random_state=42)
elif cluster_type == 'ap':
	clusters = AffinityPropagation(random_state=42)
elif cluster_type == 'db':
	clusters = DBSCAN(n_jobs=-1, eps=eps) #smaller eps more outlier
elif cluster_type == 'op':
	clusters = OPTICS(n_jobs=-1, xi=0.2)
elif cluster_type == 'sp':
	clusters = SpectralClustering(n_clusters=classes, random_state=42)
elif cluster_type == 'sb':
	clusters = SpectralBiclustering(n_clusters=classes, random_state=42)
elif cluster_type == 'sc':
	clusters = SpectralCoclustering(n_clusters=classes, random_state=42)
clusters.fit(img_features)
cluster_df = pd.DataFrame()
cluster_df["image_name"] = img_name
cluster_df["cluster_label"] = clusters.labels_
cluster_df["audio_name"] = audio_name



print('[INFO] moving files...')
if not os.path.isdir(cluster_type_dir): os.mkdir(cluster_type_dir)
os.mkdir(cluster_result_dir)
os.mkdir(cluster_result_dir+'/img')
os.mkdir(cluster_result_dir+'/audio')
os.mkdir(cluster_result_dir+'/img/-1')
os.mkdir(cluster_result_dir+'/audio/-1')
for i in range(classes):
	os.mkdir(cluster_result_dir+'/img/'+str(i))
	os.mkdir(cluster_result_dir+'/audio/'+str(i))
for i in range(len(cluster_df)):
	shutil.copy(cluster_img_path+'/'+str(cluster_df['image_name'][i]), cluster_result_dir+'/img/'+str(cluster_df['cluster_label'][i]))
	shutil.copy(cluster_audio_dir+'/'+str(cluster_df['audio_name'][i]), cluster_result_dir+'/audio/'+str(cluster_df['cluster_label'][i]))

cluster_color = []
manual_color = []
tsne_x = []
tsne_y = []
manual_label = []
for i in range(img_features_scatter.shape[0]):
	cluster_color.append(color_list[cluster_df["cluster_label"][i]])
	tsne_x.append(img_features_scatter[i,0])
	tsne_y.append(img_features_scatter[i,1])


print('[INFO] visualizing...')
font_size = 5
fig, ax = plt.subplots(figsize=(8,6))
for i in range(len(tsne_x)):
	ax.text(tsne_x[i], tsne_y[i], cluster_label[i], 
			color=cluster_color[i], fontdict={'weight': 'bold', 'size': font_size})
ax.set_title('outlier removal clustering')
ax.set_axis_off()
fig.suptitle(model_name+'_'+fea_type+'_'+cluster_type)
fig.savefig(cluster_plot_name)