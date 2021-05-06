# -*- coding: utf-8 -*-
"""
Created on Wed May  5 22:12:11 2021

@author: samuel
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow.keras.layers import  Dense,Dropout,Input,InputLayer,Conv2D,UpSampling2D,DepthwiseConv2D
from tensorflow.keras.layers import Flatten,MaxPooling2D,Conv2DTranspose,AveragePooling2D

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model,Sequential

from tensorflow.keras.optimizers import Adam


from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import img_to_array,load_img

from PIL import Image
from tensorflow.keras.utils import plot_model
from math import ceil

import random
import cv2
from skimage import io,color

tf.compat.v1.disable_eager_execution()
 
filenames = random.sample(os.listdir('C:/Users/samue/Desktop/image colorisation/emilwallner-datasets-colornet-2/images/Train'),500)
lspace=[]
abspace=[]

for file in filenames:
    rgb=io.imread("emilwallner-datasets-colornet-2/images/Train/"+file)
    lab_image = cv2.cvtColor(rgb,cv2.COLOR_BGR2LAB)
    l_channel,a_channel,b_channel =cv2.split(lab_image)
    lspace.append(l_channel)
    replot_lab = np.zeros((256,256,2))
    replot_lab[:,:,0]=a_channel
    replot_lab[:,:,1]= b_channel
    abspace.append(replot_lab)
    transfer = cv2.merge([l_channel,a_channel,b_channel])
    transfer = cv2.cvtColor(transfer.astype("uint8"),cv2.COLOR_LAB2LBGR)
lspace=np.asarray(lspace)
abspace=np.asarray(abspace)


X=lspace
Y=abspace


#creating model vgg+cnn
model6 = VGG16(weights ="imagenet",include_top=False,input_shape=(256,256,3))
model6.layers[0].trainable = False
model =Sequential()
model.add(InputLayer(input_shape=(X.shape[1],X.shape[2],1)))
model.add(layers.Dense(units=3))
model.add(Model(inputs=model6.inputs,outputs=model6.layers[-10].output))
model.add(UpSampling2D((2,2)))   
model.add(UpSampling2D((2,2)))   
model.add(DepthwiseConv2D(32,(2,2),activation='tanh',padding='same'))
model.add(UpSampling2D((2,2))) 
model.add(DepthwiseConv2D(32,(2,2),activation='tanh',padding='same'))
model.add(layers.ReLU(0.3))
model.add(layers.Dropout(0.4))
model.add(UpSampling2D((2,2)))   
model.add(UpSampling2D((2,2))) 
model.add(DepthwiseConv2D(32,(2,2),activation='tanh',padding='same'))
model.add(layers.ReLU(0.3))
model.add(layers.Dropout(0.2))
model.add(UpSampling2D((2,2))) 
model.add(layers.ReLU(0.3))
model.add(layers.Dropout(0.2))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(layers.Dense(units=2))
print(model.summary())


def adam_optimiser():
    return Adam(lr=0.001,beta_1=0.99,beta_2=0.999)

model.compile(loss='mape',optimizer=adam_optimiser())


#data preparation

X=((X.reshape(X.shape[0],X.shape[1],X.shape[2],1)))
X=(X-255)/255
Y=(Y-255)/255

trainsize=ceil(0.8*X.shape[0])
testsize=ceil(0.2*X.shape[0])+1

train_inp=X[:trainsize,]
test_inp=X[testsize:,]

train_out=Y[:trainsize,]
test_out=Y[testsize:,]


model.fit(x=train_inp,y=train_out,batch_size=1,epochs=5)




train_pred = model.predict(train_inp[:1])
test_pred =model.predict(test_inp[2:3])

train_random = random.randint(1,trainsize)
test_random = random.randint(1,testsize)

check = np.interp(train_pred,(train_pred.min(),train_pred.max()),(0,255))
check1 = np.interp(test_pred,(test_pred.min(),test_pred.max()),(0,255))

l_channel =test_inp[1]*255
a_channel = check1[1,:,:,0]
b_channel = check1[1,:,:,1]

transfer = cv2.merge([l_channel,a_channel,b_channel])
transfer = cv2.cvtColor(transfer.astype("uint8"),cv2.COLOR_LAB2LBGR)

plt.imshow(transfer)
