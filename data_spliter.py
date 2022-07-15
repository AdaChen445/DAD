import os, glob
import numpy as np
import shutil
import random
from tqdm import tqdm

input_dir = '../ng2ok_SM'
val_persent = 0.2


val_dir = input_dir+'/val'
train_test_dir = input_dir+'/test_train'
os.mkdir(val_dir)
os.mkdir(train_test_dir)

filenames = glob.glob(os.path.join(input_dir, '*.png'))
total = len(filenames)
val_part = random.choices(filenames, k=int(total*val_persent))

for filename in tqdm(val_part):
	shutil.copy(filename, val_dir)

for filename in tqdm(filenames):
	if filename in val_part: continue
	shutil.copy(filename, train_test_dir)