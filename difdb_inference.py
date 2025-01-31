import time
import os, sys
import numpy as np
import multiprocessing
import tensorflow_datasets as tfds
import tensorflow as tf
import requests
import cv2

from model_inference import *
from datasets import load_dataset
from randimage import get_random_image
from tqdm import tqdm
from itertools import repeat
from concurrent.futures import ThreadPoolExecutor
from numpy import asarray

model = "MobileNetV2_fashion_mnist.tflite"
num_class = 10

# input image size
img_size = (96, 96)

# use split='train[0:50000] to get difdb_imgs_array_100k_1.npy
dataset = load_dataset('poloclub/diffusiondb', '2m_first_100k', split='train[50000:100000]')
data = dataset['image']
x = [np.array(i.resize(img_size)) for i in data]
x = np.asarray(x)
print(x.shape)
with open('difdb_imgs_array_100k_2.npy', 'wb') as f:
    np.save(f, x)

# check the range of pixel values of raw images 
x_inference_1 = np.load('difdb_imgs_array_100k_1.npy')
x_inference_2 = np.load('difdb_imgs_array_100k_2.npy')
x_inference = np.concatenate((x_inference_1, x_inference_2), axis=0)
y_inference = np.zeros(x_inference.shape[0])
print(x_inference.shape)
print("raw img range:", np.min(x_inference), "-", np.max(x_inference))

# normalize input to [-1,1]
if model.split('_')[0] == 'MobileNetV2':
    print('Applying MobileNetV2 preprocess')
    normalized_x = tf.keras.applications.mobilenet_v2.preprocess_input(x_inference)
elif model.split('_')[0] == 'InceptionV3':
    print('Applying InceptionV3 preprocess')
    normalized_x = tf.keras.applications.inception_v3.preprocess_input(x_inference)
elif model.split('_')[0] == 'EfficientNetV2':
    print('Applying EfficientNetV2 preprocess')
    normalized_x = tf.keras.applications.inception_v3.preprocess_input(x_inference)

print("normalized image range:", np.min(normalized_x), "-", np.max(normalized_x))

normalized_x = normalized_x.astype(np.float32)
y_inference = y_inference.astype(np.float32)

y_inference = inference_synthesis(normalized_x, y_inference, model_name=model, batch_size=100, num_class=num_class)
print("target model prediction info:")
print(np.unique(y_inference, return_counts=True))

# save inference data and its prediction
with open('/data/difdb_normalized_x_' + model.split('.')[0] + '.npy', 'wb') as f:
    np.save(f, normalized_x)

with open('/data/difdb_normalized_y_' + model.split('.')[0] + '.npy', 'wb') as f:
    np.save(f, y_inference)
