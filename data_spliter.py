import os, glob
import numpy as np
import shutil
import random
from tqdm import tqdm

input_dir = '../ok_SM'
val_persent = 0.2
ok_train_test_num = 3000


val_dir = input_dir+'/val'
train_test_dir = input_dir+'/test_train'
os.mkdir(val_dir)
os.mkdir(train_test_dir)
filenames = glob.glob(os.path.join(input_dir, '*.png'))
total = len(filenames)


### ng and ng2ok
# val_part = random.choices(filenames, k=int(total*val_persent))
# for filename in tqdm(val_part):
# 	shutil.copy(filename, val_dir)
# for filename in tqdm(filenames):
# 	if filename in val_part: continue
# 	shutil.copy(filename, train_test_dir)


### ok
train_test_part = random.choices(filenames, k=ok_train_test_num)
for filename in tqdm(train_test_part):
	shutil.copy(filename, train_test_dir)
for filename in tqdm(filenames):
	if filename in train_test_part: continue
	shutil.move(filename, val_dir)