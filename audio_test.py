import numpy as np
import librosa
from scipy.signal import butter, lfilter
import sys

### LPF and output sound
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

y, sr = librosa.load('test_meterial/221.wav', sr=None)
y = butter_lowpass_filter(y, 10000, sr, 8)
y = butter_highpass_filter(y, 2000, sr, 8)
y = butter_bandstop_filter(y, 4000, 10000, sr, 8)
import soundfile as sf
sf.write('ttout.wav', y, sr, 'PCM_24')
from playsound import playsound
playsound('ttout.wav')