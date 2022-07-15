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
ap.add_argument("-t", required=True)
ap.add_argument("-f", required=True)
ap.add_argument("-c", required=False)
ap.add_argument("-mode", required=True)
ap.add_argument("-model", required=True)
ap.add_argument('-eps')
args = vars(ap.parse_args())
label_type = str(args['t']) #ori/new
fea_type = str(args['f']) #spectrogram/specMfcc/specChroma/specMfccChroma/melSpec/melChroma or others
mode = str(args['mode']) #cluster/tsne
model_name = str(args['model']) #inseption/xception/dense121/dense201/resnet50/resnet50v2/vgg16
cluster_type = str(args['c'])  #km/ap/ac/db/op/sp/sb/sc
eps = int(args['eps'])

#########arguments##########
cluster_img_path = '../features/'+fea_type
cluster_type_dir = '../'+cluster_type
# cluster_result_dir = cluster_type_dir+'/'+model_name+'_'+fea_type+'_clusterResult'
cluster_result_dir = cluster_type_dir+'/'+eps
tsne_plot_name = model_name+'_'+label_type+'Label_'+fea_type
cluster_plot_name = model_name+'_clusterLabel_'+fea_type
# cluster_audio_dir = '../for_cluster_audio'
cluster_audio_dir = '../ok_audio'
cluster_num = 20
resample_num = 3000
#########arguments##########


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def modify_contrast_and_brightness(img):
	# 公式： Out_img = alpha*(In_img) + beta
	# alpha: alpha參數 (>0)，表示放大的倍数 (通常介於 0.0 ~ 3.0之間)，能夠反應對比度
	# a>1時，影象對比度被放大， 0<a<1時 影象對比度被縮小。
	# beta:  beta参数，用來調節亮度，b>0 時亮度增強，b<0 時亮度降低。
	array_alpha = np.array([3.0]) # contrast 
	array_beta = np.array([-1.0]) # brightness
	img = cv2.add(img, array_beta)
	img = cv2.multiply(img, array_alpha)
	img = np.clip(img, 0, 255)
	return img

def image_feature(img_path,model_name):
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
	if len(img_path) > 30000: img_path = random.choices(img_path, k=resample_num) #resample
	for i in tqdm(img_path):
		fname=cluster_img_path+'/'+i
		img = cv2.imread(fname)
		img = cv2.resize(img, (input_size, input_size))
		#######CV preprocessing#######
		# img = cv2.fastNlMeansDenoisingColored(img,None,8,10,7,21)
		# img = modify_contrast_and_brightness(img)
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
		else:
			x=input_preprocessing(x)
			feat=model.predict(x)
			feat=feat.flatten()

		# feat=x.flatten() #use image straight as feature

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

if mode == 'tsne':
	print('[INFO] visualizing...')
	### 2d scatter plot accroding to menual label
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

	### 3d scatter plot accroding to menual label
	# fig = plt.figure()
	# ax = Axes3D(fig)
	# img_features_scatter = TSNE(n_components=3, init='random').fit_transform(img_features)
	# img_features_scatter = scale_minmax(img_features_scatter)
	# ax.scatter(img_features_scatter[:,0], img_features_scatter[:,1], img_features_scatter[:,2], 
	# 		c=plt.cm.Set1(label[:]))
	# plt.show()

elif mode == 'cluster':
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
	### make classes folder & copy to result folder
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

	### save data to csv for cluster_merge.py
	cluster_color = []
	manual_color = []
	tsne_x = []
	tsne_y = []
	manual_label = []
	for i in range(img_features_scatter.shape[0]):
		cluster_color.append(color_list[cluster_df["cluster_label"][i]])
		manual_color.append(color_list[label[i]])
		tsne_x.append(img_features_scatter[i,0])
		tsne_y.append(img_features_scatter[i,1])
		manual_label.append(label[i])

	cluster_df['cluster_color'] = cluster_color
	cluster_df['manual_color'] = manual_color
	cluster_df['tsne_x'] = tsne_x
	cluster_df['tsne_y'] = tsne_y
	cluster_df['manual_label'] = manual_label
	cluster_df.to_csv(cluster_result_dir+'/cluster_df.csv')



	# print('[INFO] cluster evaluating...')
	# ### auto eval how good the cluster result is
	# f_report = open(cluster_type+'/'+'cluster_eval_report.txt', 'a')
	# f_ok = open(cluster_result_dir+'/ok_label.txt', 'w')
	# cluster_img_folder = os.listdir(cluster_result_dir+'/img')
	# eval_point=10
	# for cluster_id in cluster_img_folder:
	# 	ok_num = 0
	# 	ng_num = 0
	# 	for img_name in os.listdir(cluster_result_dir+'/img/'+cluster_id):
	# 		name_list = img_name.split('_')
	# 		if name_list[1] in ('1', '2', '3', '1loud', 'pulse', 'good', 'goodN2O'): ok_num = ok_num+1
	# 		else: ng_num = ng_num+1
	# 	if ok_num > ng_num:
	# 		f_ok.write(cluster_id+'\n')
	# 		if ng_num > ok_num*0.05: eval_point = eval_point-1
	# 	else:
	# 		if ok_num > ng_num*0.3: eval_point = eval_point-1

	# print(model_name+'_'+fea_type+' eval point: '+str(eval_point))
	# f_report.write(model_name+'_'+fea_type+': '+str(eval_point)+'\n')
	# f_report.close()
	# f_ok.close()

	


	print('[INFO] visualizing...')
	clester_df = pd.read_csv(cluster_result_dir+'/cluster_df.csv')
	tsne_x = clester_df['tsne_x']
	tsne_y = clester_df['tsne_y']
	manual_label = clester_df['manual_label']
	cluster_label = clester_df['cluster_label']
	manual_color = clester_df['manual_color']
	cluster_color = clester_df['cluster_color']
	### outside of folder
	# plot_name = '../'+cluster_type+'/'+model_name+'_'+fea_type+'_'+cluster_type+'_compare'
	### inside of folder
	plot_name = cluster_result_dir+'/'+model_name+'_'+fea_type+'_'+cluster_type+'_compare'

	### tsne accroding to menual label
	# ok_label = []
	# with open(cluster_result_dir+'/ok_label.txt', 'r') as f:
	# 	for line in f.readlines():
	# 		ok_label.append(int(line))
	# font_size = 5
	# fig, axes = plt.subplots(2,2,figsize=(8,6))
	# for i in range(len(tsne_x)):
	# 	axes[0,0].text(tsne_x[i], tsne_y[i], manual_label[i], 
	# 			color=manual_color[i], fontdict={'weight': 'bold', 'size': font_size})
	# 	if manual_label[i] in (1, 2, 3, 4, 11, 13):
	# 		axes[0,1].text(tsne_x[i], tsne_y[i], manual_label[i], 
	# 			color='blue', fontdict={'weight': 'bold', 'size': font_size})
	# 	else:
	# 		axes[0,1].text(tsne_x[i], tsne_y[i], manual_label[i], 
	# 			color='red', fontdict={'weight': 'bold', 'size': font_size})
	# axes[0,0].set_title('manual label 13 classes')
	# axes[0,1].set_title('manual label NG / OK')
	# axes[0,0].set_axis_off()
	# axes[0,1].set_axis_off()
	# ### tsne according to cluster result
	# font_size = 5
	# for i in range(len(tsne_x)):
	# 	axes[1,0].text(tsne_x[i], tsne_y[i], cluster_label[i], 
	# 			color=cluster_color[i], fontdict={'weight': 'bold', 'size': font_size})
	# 	if cluster_label[i] in ok_label:
	# 		axes[1,1].text(tsne_x[i], tsne_y[i], cluster_label[i], 
	# 			color='blue', fontdict={'weight': 'bold', 'size': font_size})
	# 	else:
	# 		axes[1,1].text(tsne_x[i], tsne_y[i], cluster_label[i], 
	# 			color='red', fontdict={'weight': 'bold', 'size': font_size})
	# axes[1,0].set_title('cluster label 20 classes')
	# axes[1,1].set_title('cluster label NG / OK')
	# axes[1,0].set_axis_off()
	# axes[1,1].set_axis_off()
	# fig.suptitle(model_name+'_'+fea_type+'_'+cluster_type)
	# fig.savefig(plot_name)

	### for stage1
	font_size = 5
	fig, axes = plt.subplots(2,1,figsize=(8,6))
	for i in range(len(tsne_x)):
		axes[0].text(tsne_x[i], tsne_y[i], manual_label[i], 
				color=manual_color[i], fontdict={'weight': 'bold', 'size': font_size})
	axes[0].set_title('ok')
	axes[0].set_axis_off()
	### tsne according to cluster result
	font_size = 5
	for i in range(len(tsne_x)):
		axes[1].text(tsne_x[i], tsne_y[i], cluster_label[i], 
				color=cluster_color[i], fontdict={'weight': 'bold', 'size': font_size})
	axes[1].set_title('outlier removal clustering')
	axes[1].set_axis_off()
	fig.suptitle(model_name+'_'+fea_type+'_'+cluster_type)
	fig.savefig(plot_name)