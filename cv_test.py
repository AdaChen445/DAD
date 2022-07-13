import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.exposure import equalize_hist
import librosa
from scipy.signal import butter, lfilter
import sys
import random

ori = cv2.imread('test_meterial/tt.png', cv2.IMREAD_GRAYSCALE)/255

## data types
# print('###########')
# y = np.array(img, dtype=np.uint8)
# y = cv2.resize(y, (224, 224))
# # y = equalize_hist(y)
# print(y)
# print(sys.getsizeof(y)/(1024**2)) #MB

# print('###########')
# y = np.array(img, dtype=np.float32)
# # y = cv2.resize(y, (224, 224))
# y = equalize_hist(y)
# print(y)
# print(sys.getsizeof(y))

# print('###########')
# y = np.array(img, dtype=np.float32) / 255.0
# print(y)
# print(sys.getsizeof(y))

## CV
# img = cv2.fastNlMeansDenoisingColored(img,None,8,10,7,21)
# def modify_contrast_and_brightness(img):
# 	# 公式： Out_img = alpha*(In_img) + beta
# 	# alpha: alpha參數 (>0)，表示放大的倍数 (通常介於 0.0 ~ 3.0之間)，能夠反應對比度
# 	# a>1時，影象對比度被放大， 0<a<1時 影象對比度被縮小。
# 	# beta:  beta参数，用來調節亮度
# 	# 常數項 beta 用於調節亮度，b>0 時亮度增強，b<0 時亮度降低。
# 	array_alpha = np.array([2.5]) # contrast 
# 	array_beta = np.array([-1.0]) # brightness
# 	# add a beta value to every pixel 
# 	img = cv2.add(img, array_beta)
# 	# multiply every pixel value by alpha
# 	img = cv2.multiply(img, array_alpha)
# 	img = np.clip(img, 0, 255)
# 	return img
# def modify_contrast_and_brightness2(img, brightness=0 , contrast=100):
# 	import math
# 	brightness = 0
# 	contrast = +100 # - 減少對比度/+ 增加對比度
# 	B = brightness / 255.0
# 	c = contrast / 255.0 
# 	k = math.tan((45 + 44 * c) / 180 * math.pi)
# 	img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)
# 	img = np.clip(img, 0, 255).astype(np.uint8)
# 	return img
# img = modify_contrast_and_brightness(img)
# img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2

def add_verti_hori_gaussian_noise(img):
	verti_start = int(np.random.rand()*img.shape[1]*0.9)
	hori_start = int(np.random.rand()*img.shape[0]*0.9)
	verti_end = int(verti_start + np.random.rand()*img.shape[1]*0.1)
	hori_end = int(hori_start + np.random.rand()*img.shape[0]*0.1)
	verti_pos = [img.shape[0], verti_start, 0, verti_end]
	hori_pos = [hori_start, img.shape[1], hori_end, 0]
	def add_gaussian_noise(img, pos):
	    noise = np.random.normal(loc=0, scale=0.3, size=img.shape)
	    for i in range(pos[0]):
	    	for j in range(pos[1]):
	    		noise[i,j]=0
	    for i in range(int(img.shape[0]-pos[2])):
	    	for j in range(int(img.shape[1]-pos[3])):
	    		noise[pos[2]+i, pos[3]+j]=0
	    img = img + noise
	    gaussian_out = np.clip(img, 0, 1)
	    return gaussian_out
	img = add_gaussian_noise(img, verti_pos)
	img = add_gaussian_noise(img, hori_pos)
	return img


img = add_verti_hori_gaussian_noise(ori)

plt.subplot(121),plt.imshow(ori)
plt.subplot(122),plt.imshow(img)
plt.show()
# cv2.imwrite('ttout.png', img)
