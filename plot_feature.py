import matplotlib.pyplot as plt
import librosa
import librosa.display
import os, glob
import numpy as np
import skimage.io
from scipy.signal import butter, lfilter
import argparse
import cv2
import random
from tqdm import tqdm
from skimage.exposure import equalize_adapthist
from skimage.exposure import rescale_intensity
from skimage.exposure import equalize_hist


def scale_minmax(X, min=0.0, max=1.0):
	X_std = (X - X.min()) / (X.max() - X.min())
	X_scaled = X_std * (max - min) + min
	return X_scaled

def butter_lowpass_filter(data, cutoff, fs, order):
	nyq = 0.5 * fs
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='low', analog=False)
	y = lfilter(b, a, data)
	return y

def butter_highpass_filter(data, cutoff, fs, order):
	nyq = 0.5 * fs
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='high', analog=False)
	y = lfilter(b, a, data)
	return y

def butter_bandstop_filter(data, under, upper, fs, order):
	nyq = 0.5 * fs
	stop_band = (under/nyq, upper/nyq)
	b, a = butter(order, stop_band, btype='bandstop', analog=False)
	y = lfilter(b, a, data)
	return y

def add_verti_hori_gaussian_noise(img):
	img = img/255.0
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
	img = np.array(img*255, dtype=np.uint8)
	return img

###

def spectrum(y, sr, out_name, hop_length):
	###frame by frame
	# sextion_index = 0
	# for i in range(int(len(y)/hop_length)):
	# 	fig, ax = plt.subplots()
	# 	section = y[sextion_index:(sextion_index+hop_length*2)]
	# 	D = np.abs(librosa.stft(y=section, n_fft=hop_length*2,  hop_length=hop_length))
	# 	ax.plot(D)
	# 	freqs = librosa.fft_frequencies(sr=sr, n_fft=hop_length*2)
	# 	plt.loglog(freqs, np.mean(mag**2, axis=1)/(Nfft/2)**2)
	# 	ax.set_yscale("log")
	# 	out_name.replace('.png','')
	# 	fig.savefig(str(i)+'_'+out_name)
	# 	sextion_index = sextion_index+hop_length

	###whole frame
	fig, ax = plt.subplots()
	N = len(y) #204800
	n = np.arange(N)
	T = N/sr
	freq = n/T
	from scipy.fft import fft
	Y = fft(y)
	### for visualize
	# ax.plot(freq, np.abs(Y))
	# ax.set_xlabel('Feeq(Hz)')
	# ax.set_ylabel('Amplitude')
	# ax.axis(ymin=0,ymax=1000)
	# ax.axis(xmin=0,xmax=25000)
	# # ax.set_xscale("log")
	# # ax.set_yscale("log")
	# fig.savefig(out_name)

	img = scale_minmax(img, 0, 255).astype(np.uint8)
	return np.array([img,img,img,img])

def spectral_contrast(y, sr, out_name, hop_length):
	### for training
	D = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length)
	img = scale_minmax(D, 0, 255).astype(np.uint8)
	img = 255-img
	return img

	### for visualize
	# fig, ax = plt.subplots()
	# D = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length)
	# librosa.display.specshow(D,ax=ax, sr=sr, hop_length=hop_length, x_axis='time')
	# # plt.colorbar(format='%+2.0f dB', ax=ax)
	# fig.savefig(out_name)

def tonnetz(y, sr, out_name):
	### for training
	D = librosa.feature.tonnetz(y=y, sr=sr)
	img = scale_minmax(D, 0, 255).astype(np.uint8)
	img = 255-img
	return img

	### for visualize
	# fig, ax = plt.subplots()
	# D = librosa.feature.tonnetz(y=y, sr=sr)
	# librosa.display.specshow(D,ax=ax, sr=sr, y_axis='tonnetz', x_axis='time')
	# # plt.colorbar(format='%+2.0f dB', ax=ax)
	# fig.savefig(out_name)

def mel(y, sr, out_name, hop_length):
	### for training
	mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=hop_length*2, hop_length=hop_length)
	mels = np.log(mels + 1e-9) # add small number to avoid log(0)
	img = scale_minmax(mels, 0, 255).astype(np.uint8) # scale to fit inside 8-bit range
	img = np.flip(img, axis=0) # put low frequencies at the bottom in image
	img = 255-img # make black = more energy
	return img
	
	### for visualize
	# D = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=hop_length*2, hop_length=hop_length)
	# D = librosa.power_to_db(D, ref=np.max)
	# fig, ax = plt.subplots()
	# librosa.display.specshow(D, x_axis='time', y_axis='mel', sr=sr, ax=ax)
	# fig.savefig(out_name)

def mfcc(y, sr, out_name, hop_length):
	### for training
	D = librosa.feature.mfcc(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length, n_mfcc=40, dct_type=3,norm='ortho')
	img = scale_minmax(D, 0, 255).astype(np.uint8)
	img = 255-img
	return img

	### for visualize
	# fig, ax = plt.subplots()
	# D = librosa.feature.mfcc(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length)
	# librosa.display.specshow(D,ax=ax, sr=sr, hop_length=hop_length, x_axis='time')
	# # plt.colorbar(format='%+2.0f dB', ax=ax)
	# fig.savefig(out_name)

def chroma(y, sr, out_name, hop_length):
	### for training
	D = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length)
	# D = librosa.feature.chroma_cqt(y=y, sr=sr)
	# D = librosa.feature.chroma_cens(y=y, sr=sr)
	img = scale_minmax(D, 0, 255).astype(np.uint8)
	img = 255-img
	return img

	### for visualize
	# fig, ax = plt.subplots()
	# D = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length)
	# librosa.display.specshow(D,ax=ax, sr=sr, hop_length=hop_length, x_axis='time', y_axis='chroma')
	# # plt.colorbar(format='%+2.0f dB', ax=ax)
	# fig.savefig(out_name)

def spectrogram(y, sr, out_name, hop_length, log):
	### for training
	D = np.abs(librosa.stft(y=y, n_fft=hop_length*2, hop_length=hop_length))
	# D = np.abs(librosa.stft(y=y, n_fft=hop_length*2, hop_length=hop_length))
	D = np.log(D + 1e-9)
	# D = equalize_hist(D)
	img = scale_minmax(D, 0, 255).astype(np.uint8)
	img = np.flip(img, axis=0) 
	img = 255-img 
	img = img[313:473, :] #160 #original size 513*401
	# img = img[0:473, :] 
	return img

	### for visualize
	# fig, ax = plt.subplots()
	# D = np.abs(librosa.stft(y=y, n_fft=hop_length*2,  hop_length=hop_length))
	# D = librosa.amplitude_to_db(D, ref=np.max)
	# if log: yaix = 'log' 
	# else: yaix = 'linear'
	# librosa.display.specshow(D,ax=ax, sr=sr, hop_length=hop_length, x_axis='time', y_axis=yaix)
	# ax.axis(ymin=0,ymax=10000) #only need feature below 10k
	# # plt.colorbar(format='%+2.0f dB', ax=ax)
	# fig.savefig(out_name)

###

def melspec(y, sr, out_name, hop_length):
	D1 = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=hop_length*2, hop_length=hop_length)
	D1 = np.log(D1 + 1e-9)
	D1 = scale_minmax(D1, 0, 255).astype(np.uint8)
	D1 = np.flip(D1, axis=0)
	D1 = 255-D1

	D2 = np.abs(librosa.stft(y=y, n_fft=hop_length*2,  hop_length=hop_length))
	D2 = np.log(D2 + 1e-9)
	D2 = scale_minmax(D2, 0, 255).astype(np.uint8)
	D2 = np.flip(D2, axis=0)
	D2 = 255-D2
	D2 = D2[313:473, :]

	img = np.concatenate((D1,D2), axis=0)
	return img

def specsc(y, sr, out_name, hop_length):
	D1 = np.abs(librosa.stft(y=y, n_fft=hop_length*2,  hop_length=hop_length))
	D1 = np.log(D1 + 1e-9)
	D1 = scale_minmax(D1, 0, 255).astype(np.uint8)
	D1 = np.flip(D1, axis=0)
	D1 = 255-D1
	D1 = D1[313:473, :]

	D2 = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length)
	D2 = scale_minmax(D2, 0, 255).astype(np.uint8)
	D2 = 255-D2

	img = np.concatenate((D1,D2), axis=0)
	return img

def spectonnetz(y, sr, out_name, hop_length):
	D1 = np.abs(librosa.stft(y=y, n_fft=hop_length*2,  hop_length=hop_length))
	D1 = np.log(D1 + 1e-9)
	D1 = scale_minmax(D1, 0, 255).astype(np.uint8)
	D1 = np.flip(D1, axis=0)
	D1 = 255-D1
	D1 = D1[313:473, :]

	D2 = librosa.feature.tonnetz(y=y, sr=sr)
	D2 = scale_minmax(D2, 0, 255).astype(np.uint8)
	D2 = 255-D2

	img = np.concatenate((D1,D2), axis=0)
	return img

def specmfccsc(y, sr, out_name, hop_length):
	D1 = np.abs(librosa.stft(y=y, n_fft=hop_length*2,  hop_length=hop_length))
	D1 = np.log(D1 + 1e-9)
	D1 = scale_minmax(D1, 0, 255).astype(np.uint8)
	D1 = np.flip(D1, axis=0)
	D1 = 255-D1
	D1 = D1[313:473, :]

	D2 = librosa.feature.mfcc(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length)
	D2 = scale_minmax(D2, 0, 255).astype(np.uint8)
	D2 = 255-D2

	D3 = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length)
	D3 = scale_minmax(D3, 0, 255).astype(np.uint8)
	D3 = 255-D3

	img = np.concatenate((D1,D2,D3), axis=0)
	return img

def specmfcctonnetz(y, sr, out_name, hop_length):
	D1 = np.abs(librosa.stft(y=y, n_fft=hop_length*2,  hop_length=hop_length))
	D1 = np.log(D1 + 1e-9)
	D1 = scale_minmax(D1, 0, 255).astype(np.uint8)
	D1 = np.flip(D1, axis=0)
	D1 = 255-D1
	D1 = D1[313:473, :]

	D2 = librosa.feature.mfcc(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length)
	D2 = scale_minmax(D2, 0, 255).astype(np.uint8)
	D2 = 255-D2

	D3 = librosa.feature.tonnetz(y=y, sr=sr)
	D3 = scale_minmax(D3, 0, 255).astype(np.uint8)
	D3 = 255-D3

	img = np.concatenate((D1,D2,D3), axis=0)
	return img

def melmfccchroma(y, sr, out_name, hop_length):
	D1 = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=hop_length*2, hop_length=hop_length)
	D1 = np.log(D1 + 1e-9)
	D1 = scale_minmax(D1, 0, 255).astype(np.uint8)
	D1 = np.flip(D1, axis=0)
	D1 = 255-D1

	D2 = librosa.feature.mfcc(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length)
	D2 = scale_minmax(D2, 0, 255).astype(np.uint8)
	D2 = 255-D2

	D3 = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length)
	D3 = scale_minmax(D3, 0, 255).astype(np.uint8)
	D3 = 255-D3

	img = np.concatenate((D1,D2,D3), axis=0)
	return img

def melchromatonnetz(y, sr, out_name, hop_length):
	D1 = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=hop_length*2, hop_length=hop_length)
	D1 = np.log(D1 + 1e-9)
	D1 = scale_minmax(D1, 0, 255).astype(np.uint8)
	D1 = np.flip(D1, axis=0)
	D1 = 255-D1

	D2 = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length)
	D2 = scale_minmax(D2, 0, 255).astype(np.uint8)
	D2 = 255-D2

	D3 = librosa.feature.tonnetz(y=y, sr=sr)
	D3 = scale_minmax(D3, 0, 255).astype(np.uint8)
	D3 = 255-D3

	img = np.concatenate((D1,D2,D3), axis=0)
	return img

def mfccchromatonnetz(y, sr, out_name, hop_length):
	D1 = librosa.feature.mfcc(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length, n_mfcc=40, dct_type=3,norm='ortho')
	D1 = scale_minmax(D1, 0, 255).astype(np.uint8)
	D1 = 255-D1

	D2 = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length)
	D2 = scale_minmax(D2, 0, 255).astype(np.uint8)
	D2 = 255-D2

	D3 = librosa.feature.tonnetz(y=y, sr=sr)
	D3 = scale_minmax(D3, 0, 255).astype(np.uint8)
	D3 = 255-D3

	img = np.concatenate((D1,D2,D3), axis=0)
	return img

def melspecmfccchroma(y, sr, out_name, hop_length):
	D1 = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=hop_length*2, hop_length=hop_length)
	D1 = np.log(D1 + 1e-9)
	D1 = scale_minmax(D1, 0, 255).astype(np.uint8)
	D1 = np.flip(D1, axis=0)
	D1 = 255-D1

	D2 = librosa.feature.mfcc(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length)
	D2 = scale_minmax(D2, 0, 255).astype(np.uint8)
	D2 = 255-D2

	D3 = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length)
	D3 = scale_minmax(D3, 0, 255).astype(np.uint8)
	D3 = 255-D3

	D4 = np.abs(librosa.stft(y=y, n_fft=hop_length*2,  hop_length=hop_length))
	D4 = np.log(D4 + 1e-9)
	D4 = scale_minmax(D4, 0, 255).astype(np.uint8)
	D4 = np.flip(D4, axis=0)
	D4 = 255-D4
	D4 = D4[313:473, :]

	img = np.concatenate((D1,D2,D3,D4), axis=0)
	return img

def hpss(y, sr, out_name, hop_length):
	H,P = librosa.decompose.hpss(librosa.stft(y=y, n_fft=hop_length*2,  hop_length=hop_length))
	D1 = np.abs(H)
	D1 = np.log(D1 + 1e-9)
	D1 = scale_minmax(D1, 0, 255).astype(np.uint8)
	D1 = np.flip(D1, axis=0)
	D1 = 255-D1
	D1 = D1[313:473, :] #160

	D2 = np.abs(P)
	D2 = np.log(D2 + 1e-9)
	D2 = scale_minmax(D2, 0, 255).astype(np.uint8)
	D2 = np.flip(D2, axis=0)
	D2 = 255-D2
	D2 = D2[313:473, :] #160

	img = np.concatenate((D1,D2), axis=0)
	return img

def hmfcc(y, sr, out_name, hop_length):
	H,P = librosa.decompose.hpss(librosa.stft(y=y, n_fft=hop_length*2,  hop_length=hop_length))
	D1 = np.abs(H)
	D1 = np.log(D1 + 1e-9)
	D1 = scale_minmax(D1, 0, 255).astype(np.uint8)
	D1 = np.flip(D1, axis=0)
	D1 = 255-D1
	D1 = D1[313:473, :] #160

	D2 = librosa.feature.mfcc(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length, n_mfcc=40, dct_type=3,norm='ortho')
	D2 = scale_minmax(D2, 0, 255).astype(np.uint8)
	D2 = 255-D2

	img = np.concatenate((D1,D2), axis=0)
	return img

def pmfcc(y, sr, out_name, hop_length):
	H,P = librosa.decompose.hpss(librosa.stft(y=y, n_fft=hop_length*2,  hop_length=hop_length))
	D1 = np.abs(P)
	D1 = np.log(D1 + 1e-9)
	D1 = scale_minmax(D1, 0, 255).astype(np.uint8)
	D1 = np.flip(D1, axis=0)
	D1 = 255-D1
	D1 = D1[313:473, :] #160

	D2 = librosa.feature.mfcc(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length)
	D2 = scale_minmax(D2, 0, 255).astype(np.uint8)
	D2 = 255-D2

	img = np.concatenate((D1,D2), axis=0)
	return img

def mfccchroma(y, sr, out_name, hop_length):
	D2 = librosa.feature.mfcc(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length)
	D2 = scale_minmax(D2, 0, 255).astype(np.uint8)
	D2 = 255-D2

	D3 = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length)
	D3 = scale_minmax(D3, 0, 255).astype(np.uint8)
	D3 = 255-D3

	img = np.concatenate((D2,D3), axis=0)
	return img

def specchroma(y, sr, out_name, hop_length):
	D1 = np.abs(librosa.stft(y=y, n_fft=hop_length*2,  hop_length=hop_length))
	D1 = np.log(D1 + 1e-9)
	D1 = np.flip(D1, axis=0)
	D1 = D1[313:473, :]
	D1 = scale_minmax(D1, 0, 255).astype(np.uint8)
	D1 = 255-D1

	D2 = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length)
	D2 = scale_minmax(D2, 0, 255).astype(np.uint8)
	D2 = 255-D2

	img = np.concatenate((D1,D2), axis=0)
	return img

def specmfccchroma(y, sr, out_name, hop_length):
	D1 = np.abs(librosa.stft(y=y, n_fft=hop_length*2,  hop_length=hop_length))
	D1 = np.log(D1 + 1e-9)
	D1 = np.flip(D1, axis=0)
	D1 = D1[313:473, :]
	D1 = scale_minmax(D1, 0, 255).astype(np.uint8)
	D1 = 255-D1
	D2 = librosa.feature.mfcc(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length)
	D2 = scale_minmax(D2, 0, 255).astype(np.uint8)
	D2 = 255-D2
	D3 = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length)
	D3 = scale_minmax(D3, 0, 255).astype(np.uint8)
	D3 = 255-D3
	img = np.concatenate((D1,D2,D3), axis=0)
	return img

def all_feature(y, sr, out_name, hop_length):
	fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(1,6,figsize=(30,4))
	### spectrogram
	ax1.title.set_text('spectrogram')
	D = np.abs(librosa.stft(y=y, n_fft=hop_length*2,  hop_length=hop_length))
	D = librosa.amplitude_to_db(D, ref=np.max)
	librosa.display.specshow(D,ax=ax1, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
	### mel
	ax2.title.set_text('mel')
	D = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=hop_length*2, hop_length=hop_length)
	D = librosa.power_to_db(D, ref=np.max)
	librosa.display.specshow(D, x_axis='time', y_axis='mel', sr=sr, ax=ax2)
	### mfcc
	ax3.title.set_text('mfcc')
	D = librosa.feature.mfcc(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length)
	librosa.display.specshow(D,ax=ax3, sr=sr, hop_length=hop_length, x_axis='time')
	### spectral contrast
	ax4.title.set_text('spectral contrast')
	D = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length)
	librosa.display.specshow(D,ax=ax4, sr=sr, hop_length=hop_length, x_axis='time')
	### chroma
	ax5.title.set_text('chroma')
	D = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length)
	librosa.display.specshow(D,ax=ax5, sr=sr, hop_length=hop_length, x_axis='time', y_axis='chroma')
	### tonnetz
	ax6.title.set_text('tonnetz')
	D = librosa.feature.tonnetz(y=y, sr=sr)
	librosa.display.specshow(D,ax=ax6, sr=sr, y_axis='tonnetz', x_axis='time')
	# plt.axis('off') #for training
	img = equalize_adapthist(img)
	fig.savefig(out_name,bbox_inches='tight',pad_inches = 0)

def test(y,sr):
	# D = librosa.feature.rms(y=y)
	# D = librosa.feature.zero_crossing_rate(y)
	# D = librosa.feature.spectral_centroid(y=y, sr=sr)
	# D = librosa.feature.spectral_bandwidth(y=y, sr=sr)
	# D = librosa.feature.spectral_flatness(y=y)
	# D = librosa.feature.spectral_rolloff(y=y, sr=sr)
	# D = librosa.feature.poly_features(S=np.abs(librosa.stft(y)), order=0)
	img = scale_minmax(D, 0, 255).astype(np.uint8)
	return img

###

def specmfcc(y, sr, out_name, hop_length):
	D1 = np.abs(librosa.stft(y=y, n_fft=hop_length*2, win_length=hop_length, hop_length=hop_length))
	D1 = np.log(D1 + 1e-9)
	D1 = scale_minmax(D1, 0, 255).astype(np.uint8)
	D1 = np.flip(D1, axis=0)
	D1 = 255-D1
	# D1 = D1[313:473, :] #160

	D2 = librosa.feature.mfcc(y=y, sr=sr, n_fft=hop_length*2, win_length=hop_length, hop_length=hop_length, n_mfcc=40, dct_type=3,norm='ortho')
	D2 = scale_minmax(D2, 0, 255).astype(np.uint8)
	D2 = 255-D2

	img = np.concatenate((D1,D2), axis=0)
	return img	

def melchroma(y, sr, out_name, hop_length):
	D1 = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=hop_length*2, win_length=hop_length*2)
	D1 = np.log(D1 + 1e-9)
	D1 = scale_minmax(D1, 0, 255).astype(np.uint8)
	D1 = np.flip(D1, axis=0)
	D1 = 255-D1

	D2 = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=hop_length*2, win_length=hop_length*2)
	D2 = scale_minmax(D2, 0, 255).astype(np.uint8)
	D2 = 255-D2

	img = np.concatenate((D1,D2), axis=0)
	return img

def melmfcc(y, sr, out_name, hop_length):
	D1 = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=hop_length*2, hop_length=hop_length)
	D1 = np.log(D1 + 1e-9)
	D1 = scale_minmax(D1, 0, 255).astype(np.uint8)
	D1 = np.flip(D1, axis=0)
	D1 = 255-D1

	D2 = librosa.feature.mfcc(y=y, sr=sr, n_fft=hop_length*2,  hop_length=hop_length, n_mfcc=40, dct_type=3,norm='ortho')
	D2 = scale_minmax(D2, 0, 255).astype(np.uint8)
	D2 = 255-D2

	img = np.concatenate((D1,D2), axis=0)
	return img



ap = argparse.ArgumentParser()
ap.add_argument('-l', required=True) #input folder label
ap.add_argument('-f', required=True) #specmfcc/melchroma
ap.add_argument('-o', default='temp_img_seperate')
args = vars(ap.parse_args())
label = str(args['l'])
fea_type = str(args['f'])
output_type = str(args['o'])

#########arguments##########
# fft_size=512 #801
fft_size=1024 #401
# fft_size=2048 #201
#########arguments##########

if output_type == 'features': #new 13label
	input_path = '../newLabeled/'+label
	output_path = '../features/'+fea_type

elif output_type == 'temp_img_serial': #outlier
	input_path = '../'+label+'_audio'

elif output_type == ('temp_img_seperate', 'stage1ok', 'stage1arg'):
	input_path = '../'+label+'_audio'
	output_path = '../temp_img/'+label

elif output_type == 'dcaset2_train':
	input_path = '../dcase_t2/audio/'+label+'/train'
	output_path = '../dcase_t2/image/'+label+'/train'

elif output_type == 'dcaset2_test':
	input_path = '../dcase_t2/audio/'+label+'/test'
	output_path = '../dcase_t2/image/'+label+'/test'


if not os.path.isdir('../temp_img'): os.mkdir('../temp_img')
if not os.path.isdir(output_path): os.makedirs(output_path)

filenames = glob.glob(os.path.join(input_path, '*.wav'))
for idx,filename in enumerate(tqdm(filenames)):
	fileID = filename.split(os.path.sep)[-1].replace('.wav', '.png')

	if output_type in ('features', 'temp_img_seperate', 'stage1ok', 'stage1arg', 'dcaset2_train', 'dcaset2_test'):
		out_name = output_path + '/'+label+'_' + fileID
	elif output_type == 'temp_img_serial':
		out_name = '../temp_img/'+label+'_' + str(idx) 
		idx = idx + 1


	signalData, sr = librosa.load(filename, sr=None)
	# signalData = signalData[0:sr*4]
	if fea_type == 'mel':
		img = mel(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2))
	elif fea_type == 'specMfcc':
		img = specmfcc(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2))
	elif fea_type == 'melChroma':
		img = melchroma(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2))
	elif fea_type == 'spectrogram':
		img = spectrogram(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2), log=False)
	elif fea_type == 'spectrum':
		img = spectrum(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2))
	elif fea_type == 'mfcc':
		img = mfcc(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2))
	elif fea_type == 'tonnetz':
		img = tonnetz(signalData, sr=sr, out_name=out_name)
	elif fea_type == 'chroma':
		img = chroma(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2))
	elif fea_type == 'spectral':
		img = spectral_contrast(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2))
	elif fea_type == 'all':
		all_feature(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2))
	elif fea_type == 'specChroma':
		img = specchroma(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2))
	elif fea_type == 'specMfccChroma':
		img = specmfccchroma(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2))
	elif fea_type == 'melSpec':
		img = melspec(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2))
	elif fea_type == 'melMfcc':
		img = melmfcc(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2))
	elif fea_type == 'melMfccChroma':
		img = melmfccchroma(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2))
	elif fea_type == 'melSpecMfccChroma':
		img = melspecmfccchroma(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2))
	elif fea_type == 'specsc':
		img = specsc(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2))
	elif fea_type == 'spectonnetz':
		img = spectonnetz(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2))
	elif fea_type == 'specmfccsc':
		img = specmfccsc(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2))
	elif fea_type == 'specmfcctonnetz':
		img = specmfcctonnetz(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2))
	elif fea_type == 'hpss':
		img = hpss(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2))
	elif fea_type == 'mfccchroma':
		img = mfccchroma(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2))
	elif fea_type == 'hmfcc':
		img = hmfcc(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2))
	elif fea_type == 'pmfcc':
		img = pmfcc(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2))
	elif fea_type == 'melCT':
		img = melchromatonnetz(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2))
	elif fea_type == 'mfccCT':
		img = mfccchromatonnetz(signalData, sr=sr, out_name=out_name, hop_length=int(fft_size/2))
	elif fea_type == 'test':
		img = test(signalData, sr=sr)

	if output_type == 'stage1arg':
		img = add_verti_hori_gaussian_noise(img)

	# img = cv2.resize(img,(401,200))
	skimage.io.imsave(out_name, img)