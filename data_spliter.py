import os, glob
import numpy as np
import shutil
import random
from tqdm import tqdm


import argparse
ap = argparse.ArgumentParser()
ap.add_argument('-d')
ap.add_argument('-t')
args = vars(ap.parse_args())
input_dir = str(args['d'])
input_type = str(args['t'])
# input_dir = '../ng2ok_MC'
# input_type = 'other'

val_persent = 0.2
ok_train_test_num = 3000
ok_val_num = 3000


val_dir = input_dir+'/val'
train_test_dir = input_dir+'/test_train'
os.mkdir(val_dir)
os.mkdir(train_test_dir)
filenames = glob.glob(os.path.join(input_dir, '*'))
total = len(filenames)


if input_type == 'ok':
	train_test_part = random.choices(filenames, k=ok_train_test_num)
	for filename in tqdm(train_test_part):
		try:
			shutil.move(filename, train_test_dir)
		except: continue
	val_part = []
	for filename in filenames:
		if filename in train_test_part: continue
		val_part.append(filename)
	val_part_random = random.choices(val_part, k=ok_val_num)
	for filename in tqdm(val_part_random):
		try:
			shutil.move(filename, val_dir)
		except: continue
else:
	val_part = random.choices(filenames, k=int(total*val_persent))
	for filename in tqdm(val_part):
		try:
			shutil.move(filename, val_dir)
		except: continue
	for filename in tqdm(filenames):
		if filename in val_part: continue
		shutil.move(filename, train_test_dir)


