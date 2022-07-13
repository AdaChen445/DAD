import os
import cv2
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
import time
import pickle
tf.compat.v1.disable_eager_execution()

def processing_image(img_path):
    x = cv2.imread(img_path)
    x = cv2.resize(x, (224,224))
    x = np.array(x, dtype=np.uint8)
    x = np.expand_dims(x, axis=0)# 加上 batch size
    return x


def gradcam(model, le, x):
    tstart = time.time()
    preds = model.predict(x)
    tend = time.time()
    print('pred time: '+str(tend-tstart))
    pred_class = np.argmax(preds[0])
    pred_conffident = np.max(preds[0])
    pred_class_name = le.classes_[pred_class]
    
    # 預測分類的輸出向量
    pred_output = model.output[:, pred_class]
    
    # 最後一層 convolution layer 輸出的 feature map
    # last_conv_layer = model.get_layer('block7g_project_conv')
    last_conv_layer = model.get_layer('block14_sepconv2')
    
    # 求得分類的神經元對於最後一層 convolution layer 的梯度
    grads = K.gradients(pred_output, last_conv_layer.output)[0]
    
    # 求得針對每個 feature map 的梯度加總
    pooled_grads = K.sum(grads, axis=(0, 1, 2))
    
    # K.function() 讓我們可以藉由輸入影像至 `model.input` 得到 `pooled_grads` 與
    # `last_conv_layer[0]` 的輸出值，像似在 Tensorflow 中定義計算圖後使用 feed_dict
    # 的方式。
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    
    # 傳入影像矩陣 x，並得到分類對 feature map 的梯度與最後一層 convolution layer 的 
    # feature map
    pooled_grads_value, conv_layer_output_value = iterate([x])
    
    # 將 feature map 乘以權重，等於該 feature map 中的某些區域對於該分類的重要性
    for i in range(pooled_grads_value.shape[0]):
        conv_layer_output_value[:, :, i] *= (pooled_grads_value[i])
        
    # 計算 feature map 的 channel-wise 加總
    heatmap = np.sum(conv_layer_output_value, axis=-1)
    
    return heatmap, pred_class_name, pred_conffident


def plot_heatmap(heatmap, img_path, pred_class_name, pred_conffident):
    # ReLU
    heatmap = np.maximum(heatmap, 0)
    
    # 正規化
    heatmap /= np.max(heatmap)
    
    # 讀取影像
    img = cv2.imread(img_path)
    im = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (img.shape[1], img.shape[0]))

    # 拉伸 heatmap
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    
    
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
    ax1.title.set_text('ground truth: ng_1')
    ax2.title.set_text('pred result: '+str(pred_class_name)+' '+str(pred_conffident))
    ax1.imshow(img)
    ax2.imshow(im, alpha=0.6),plt.imshow(heatmap, cmap='jet', alpha=0.4)
    plt.show()
    


# model = load_model('./effi_0.88.h5')
model = load_model('./xcept_0.8569.h5')
# model.summary()
le = pickle.loads(open('./le.pickle','rb').read())
img_path = './heatplot/ng(1).png'
img = processing_image(img_path)
heatmap, pred_class_name, pred_conffident = gradcam(model, le, img)
plot_heatmap(heatmap, img_path, pred_class_name, pred_conffident)