from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize, suppress=True)
import cv2
import os
from tqdm import tqdm
from tensorflow.keras.models import load_model
from numpy import *
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

def DSCombination(Dic1, Dic2):
    print('[1]', Dic1, Dic2)
    ## extract the frame dicernment
    sets=set(Dic1.keys()).union(set(Dic2.keys()))
    Result=dict.fromkeys(sets,0)
    
    ## Combination process
    for i in Dic1.keys():
        for j in Dic2.keys():
            print('i',i, 'j', j)
            if set(str(i)).intersection(set(str(j))) == set(str(i)):
                print('[2_0]', set(str(i)), Result[i])
                Result[i]+=Dic1[i]*Dic2[j]
                print('[2]', set(str(i)), Result[i])
            elif set(str(i)).intersection(set(str(j))) == set(str(j)):
                print('[3_0]', set(str(j)), Result[j])
                Result[j]+=Dic1[i]*Dic2[j]
                print('[3]', set(str(j)), Result[j])
    
    ## normalize the results
    f= sum(list(Result.values()))
    print(f)
    for i in Result.keys():
        Result[i] /=f
    print('[4]',Result)
    return Result

def loadDataSplit(datapath, img_size):
    imagePaths = glob.glob(os.path.join(datapath, '*.png'))
    lebal_types = len(next(os.walk(datapath))[1])
    img = []
    labels = []
    for imagePath in tqdm(imagePaths):
        label = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (img_size, img_size))
        img.append(image)
        labels.append(label)
    img = np.array(img, dtype=np.float32)/255.0
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = to_categorical(labels, lebal_types)
    (trainX, testX, trainY, testY) = train_test_split(img, labels,
        test_size=0.25, random_state=42)
    testX = preprocess_input(testX)
    return le, trainX, testX, trainY, testY

def loadDataSplit3c(datapath, img_size):
    imagePaths = glob.glob(os.path.join(datapath+'/c1', '*.png'))
    lebal_types = len(next(os.walk(datapath+'/c1'))[1])
    img = []
    labels = []
    for imagePath in tqdm(imagePaths):
        label = imagePath.split(os.path.sep)[-2]
        imagePath1 = imagePath
        imagePath2 = datapath+'/c2/'+label+'/'+imagePath.split(os.path.sep)[-1]
        imagePath3 = datapath+'/c3/'+label+'/'+imagePath.split(os.path.sep)[-1]
        img1 = cv2.imread(imagePath1, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.resize(img1,(img_size, img_size))#, interpolation=cv2.INTER_CUBIC)
        img1 = np.array(img1, dtype=np.uint8)
        img2 = cv2.imread(imagePath2, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.resize(img2,(img_size, img_size))#, interpolation=cv2.INTER_CUBIC)
        img2 = np.array(img2, dtype=np.uint8)
        img3 = cv2.imread(imagePath3, cv2.IMREAD_GRAYSCALE)
        img3 = cv2.resize(img3,(img_size, img_size))#, interpolation=cv2.INTER_CUBIC)
        img3 = np.array(img3, dtype=np.uint8)
        image = np.dstack([img1, img2, img3]).astype(np.uint8)
        img.append(image)
        labels.append(label)
    img = np.array(img, dtype=np.float32)/255.0
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = to_categorical(labels, lebal_types)
    (trainX, testX, trainY, testY) = train_test_split(img, labels,
        test_size=0.25, random_state=42)
    testX = preprocess_input(testX)
    return le, trainX, testX, trainY, testY

def array2dict(pred, le):
    # pred_dict = {str(le.classes_[0]) : pred[0], str(le.classes_[1]) : pred[1]}
    pred_dict = {'N' : pred[0], 'NO' : pred[1]}
    return pred_dict

def dict2array(conbi_dict, le):
    # conbi_array = [conbi_dict[str(le.classes_[0])], conbi_dict[str(le.classes_[1])]]
    conbi_array = [conbi_dict['N'], conbi_dict['NO']]
    return conbi_array

def arrayDScombination(preds_1, preds_2, le):
    conbi_preds = []
    for i in range(len(preds_1)):
        dict_1 = array2dict(preds_1[i], le)
        dict_2 = array2dict(preds_2[i], le)
        conbi_dict = DSCombination(dict_1, dict_2)
        conbi_array = dict2array(conbi_dict, le)
        conbi_preds.append(conbi_array)
    conbi_preds = np.array(conbi_preds)
    return conbi_preds

def myArrayDScombination(preds_1, preds_2, le):
    conbi_preds = [[],[]]
    for i in range(len(preds_1)):
        k = preds_1[i][0]*preds_2[i][1]+preds_1[i][1]*preds_2[i][0]
        co = 1/(1-k)
        conbi_preds[0].append(co*preds_1[i][0]*preds_2[i][0])
        conbi_preds[1].append(co*preds_1[i][1]*preds_2[i][1])
    conbi_preds = np.array(conbi_preds)
    conbi_preds = np.transpose(conbi_preds)
    return conbi_preds

def predThreshold(add_thre, diff_thre, preds, ty):
    pred_del = preds
    ty_del = ty
    for i in range(preds.shape[0]):
        if abs(pred_del[i][0]-pred_del[i][1])<diff_thre:
            pred_del[i][0]=0
            pred_del[i][1]=0
            ty_del[i][0]=0
            ty_del[i][1]=0
            # # pred_del = np.delete(pred_2, i, 0)
            # ty_del = np.delete(ty2, i, 0)
        if abs(pred_del[i][0]+pred_del[i][1])<add_thre:
            pred_del[i][0]=0
            pred_del[i][1]=0
            ty_del[i][0]=0
            ty_del[i][1]=0

    pred_del_drop = pred_del[~np.all(pred_del==0, axis=1)]
    ty_del_drop = ty_del[~np.all(ty_del==0, axis=1)]
    # print(ty_del)
    # print(pred_del[pred_del[:,0].argsort()])
    return pred_del_drop, ty_del_drop

def uniform_soup(path):
    soups = []
    for model_path in os.listdir(path):
        print(model_path)
        model = load_model(path+'/'+model_path)
        soup = [np.array(w) for w in model.weights]
        soups.append(soup)
    if 0 < len(soups):
        for w1, w2 in zip(model.weights, list(zip(*soups))):
            tf.keras.backend.set_value(w1, np.mean(w2, axis = 0))
    return model

def allInOne(model_dir, data_dir, img_size, BS, is3channel=True, isModelsoup=False):
    print("[INFO] loading model...")
    if isModelsoup:
        model = uniform_soup(model_dir)
    else:
        model = load_model(model_dir)
    print("[INFO] loading data...")
    if is3channel:
        le, rx, tx, ry, ty = loadDataSplit3c(data_dir,img_size)
    else:
        le, rx, tx, ry, ty = loadDataSplit(data_dir,img_size)
    print("[INFO] evaluating...")
    pred = model.predict(x=tx, batch_size=BS)
    return pred, ty


#########arguments##########
BS = 32
img_size_1 = 300
img_size_2 = 300
le = pickle.loads(open('./benchmark/le.pickle', 'rb').read())

model_dir_1 = './benchmark/specmfcc_8952.h5'
model_dir_2 = './benchmark/melchroma_9015.h5'
datasetpath_1 = '../dataset_stage2'
datasetpath_2 = '../dataset_stage2melchroma'

model3c_dir_1 = './benchmark/3cSMFFT_9088.h5'
model3c_dir_2 = './benchmark/3cMCFFT_9088.h5'
datasetpath3c_1 = '../dataset_stage2c3FFT'
datasetpath3c_2 = '../dataset_stage2c3MCFFT'

model_soup = './benchmark/model_soup'
#########arguments##########

pred_1, ty = allInOne(model3c_dir_1, datasetpath3c_1, img_size_1, BS)
# pred_2, ty = allInOne(model3c_dir_2, datasetpath3c_2, img_size_2, BS)
# pred_conbi = arrayDScombination(pred_1, pred_2, le)

x1=[] 
x2=[]
y1=[]
y2=[]
for i in range(len(ty)):
    if le.classes_[ty.argmax(axis=1)][i]=='ng':
        x1.append(pred_1[i,0])
        y1.append(pred_1[i,1])
    elif le.classes_[ty.argmax(axis=1)][i]=='ng2ok':
        x2.append(pred_1[i,0])
        y2.append(pred_1[i,1])
    else: raise
plt.title('output of XceptionNet with specmfcc')
plt.xlabel('ng score')
plt.ylabel('ng2ok score')
plt.scatter(x1, y1, s=3, c='red', alpha=0.7)
plt.scatter(x2, y2, s=3, c='blue', alpha=0.7)
plt.show()





# print('specmfcc_3channels')
# print(classification_report(,pred_1.argmax(axis=1), target_names=le.classes_, digits=5))
# print('melchroma_3channels')
# print(classification_report(ty.argmax(axis=1),pred_2.argmax(axis=1), target_names=le.classes_, digits=5))
# print('DRC_N+N^O')
# print(classification_report(ty.argmax(axis=1),pred_conbi.argmax(axis=1), target_names=le.classes_, digits=5))

# fig, (ax1,ax2,ax3) = plt.subplots(1,3)
# ax1.set_title('specmfcc_3channels')
# ax2.set_title('melchroma_3channels')
# ax3.set_title('DRC_N+N^O')
# ConfusionMatrixDisplay.from_predictions(ty.argmax(axis=1), pred_1.argmax(axis=1), display_labels=le.classes_,ax=ax1,colorbar=False, cmap='Blues')
# ConfusionMatrixDisplay.from_predictions(ty.argmax(axis=1), pred_2.argmax(axis=1), display_labels=le.classes_,ax=ax2,colorbar=False, cmap='Blues')
# ConfusionMatrixDisplay.from_predictions(ty.argmax(axis=1), pred_conbi.argmax(axis=1), display_labels=le.classes_,ax=ax3,colorbar=False, cmap='Blues')
# plt.show()


# pred_conbi_del, ty_del =  predThreshold(0.5, 0.5, pred_conbi, ty)
# print(classification_report(ty_del.argmax(axis=1),pred_conbi_del.argmax(axis=1), target_names=le.classes_, digits=5))

# pred_ms, ty = allInOne(model_soup, datasetpath3c_1, img_size_1, BS, isModelsoup=True)
# print(classification_report(ty.argmax(axis=1),pred_ms.argmax(axis=1), target_names=le.classes_, digits=5))



