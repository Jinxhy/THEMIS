import os, sys
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

from model_inference import *
from datasets import load_dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from numpy import asarray


dataset_name = "svhn"
dataset = "svhn_cropped"
img_size = (96, 96)

ds_train, ds_test = tfds.load(
    dataset,
    split=['train', 'test'], #validation for gtsrb
    shuffle_files=True,
    as_supervised=True,
    download=True,
    data_dir='datasets/' + dataset
)

if dataset_name == 'fmnist': ds_train = ds_train.map(lambda x, y: (tf.image.grayscale_to_rgb(x), y))  # for  grayscale image only
ds_train = ds_train.map(lambda x, y: (tf.image.resize(x, img_size), y))
ds_train = ds_train.map(lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y))

data_size = ds_train.__len__()
ds_train = ds_train.take(data_size)

x_train = np.zeros((data_size, img_size[0], img_size[1], 3))
y_train = np.zeros(data_size)

for i, (image, label) in enumerate(tfds.as_numpy(ds_train)):
    x_train[i] = image
    y_train[i] = label

x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)

normalized_x, _, normalized_y, _ = train_test_split(x_train, y_train, test_size=0.9, random_state=42, stratify=y_train)

print('ds data statistic:', np.unique(normalized_y, return_counts=True))

# save inference data and its prediction
with open('ds_normalized_x_' + dataset_name + '.npy', 'wb') as f:
    np.save(f, normalized_x)

with open('ds_normalized_y_' + dataset_name + '.npy', 'wb') as f:
    np.save(f, normalized_y)




