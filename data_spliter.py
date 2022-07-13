import os, glob
import numpy as np
import shutil
import random

input_dir = './ng_SM'
val_dir = input_dir+'/val'
train_test_dir = input_dir+'/test_train'
os.mkdir(val_dir)
os.mkdir(train_test_dir)


filenames = glob.glob(os.path.join(input_dir, '*.png'))
total = len(filenames)

val_part = random.choices(filenames, k=int(total*0.2))




for filename in glob.glob(os.path.join(input_path, '*.wav')):
	fileID = filename.replace('\\', '/')
	

	try:
		signalData = librosa.load(filename, sr=16000, mono=True, dtype=np.float32)[0]
		if len(signalData.shape) == 1:
			signalData = signalData.reshape(1, -1)
		if signalData.shape != (1,64000):
			shutil.move(fileID, move_path)
			print(signalData.shape)

	except Exception as e:
		print(e)
		print(fileID)
		shutil.move(fileID, move_path)
		pass