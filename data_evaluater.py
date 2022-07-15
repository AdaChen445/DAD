from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
import cv2
from tqdm import tqdm
from tensorflow.keras.models import load_model
import os, glob
import numpy as np
import shutil
import random
from tqdm import tqdm


val_ng_dir = '../ng_SM/val'
val_n2o_dir = '../ng2ok_SM/val'
val_ok_dir = '../ok_SM/val'

val_model_name = 'xcep_tl100_n2ocleaner_db0.21'
model = load_model('./log/'+val_model_name+'/best_0.9936.h5')
le = pickle.loads(open('./log/'+val_model_name+'/le.pickle', 'rb').read())







filenames = glob.glob(os.path.join(val_dir, '*.png'))


for filename in tqdm(filenames):
	
