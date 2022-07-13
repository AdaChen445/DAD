import librosa
import os, glob
import numpy as np
import shutil

input_path = './source_test'
move_path = './bug_file'


total = str(len(glob.glob(os.path.join(input_path, '*.wav'))))
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