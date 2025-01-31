import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import sys
import time

from model_inference import *
from tqdm import tqdm


def prep_fn(img):
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) * 2
    return img


scenario = "dm"
ds_name = "fmnist"
model = "MobileNetV2_fmnist.tflite"

# for 5000 (cifar10), 6000 (fmnist) and stl10
if ds_name == "cifar10":
    num_sample_per_class = 5000
elif ds_name == "fmnist":
    num_sample_per_class = 6000

num_sample_per_class_gtsrb_list = [168, 1176, 1608, 1056, 1680, 1728, 624, 504, 336, 888, 960,
                                   1776, 168, 288, 264, 312, 408, 216, 1200, 480, 192, 432,
                                   1800, 216, 360, 624, 192, 551, 336, 960, 312, 168, 1656,
                                   1128, 240, 288, 192, 192, 1584, 1488, 336, 1152, 1128]
num_sample_per_class_svhn_list = [4948, 13861, 10585, 8497, 7458, 6882, 5727, 5595, 5045, 4659]
num_synthetic_samples = 10000

if scenario == "dm":
    normalized_x = np.load('/data/difdb_normalized_x_' + model.split('.')[0] + '.npy')
    # normalized_x = (normalized_x + 1) * 127.5
    normalized_y = np.load('/data/difdb_normalized_y_' + model.split('.')[0] + '.npy')
elif scenario == "ds":
    normalized_x = np.load('/data/ds_normalized_x_' + ds_name + '.npy')
    normalized_y = np.load('/data/ds_normalized_y_' + ds_name + '.npy')

print('inference data pixel range:', np.min(normalized_x), np.max(normalized_x))
labels, counts = np.unique(normalized_y, return_counts=True)
print("inference data prediction:", labels, counts)

if (ds_name == "cifar10") or (ds_name == "fmnist"):
    synthesis_counts = [(l, num_sample_per_class - c) for l, c in
                        zip(labels, counts)]  # for cifar-10, fmnist, and stl10
    num_class = 10
elif ds_name == "gtsrb":
    synthesis_counts = [(l, num_sample_per_class_gtsrb_list[int(l)] - c) for l, c in zip(labels, counts)]  # for gtsrb
    num_class = 43
elif ds_name == "svhn":
    synthesis_counts = [(l, num_sample_per_class_svhn_list[int(l)] - c) for l, c in zip(labels, counts)]  # for svhn
    num_class = 10

print("synthesis data size per label:", synthesis_counts)

generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=50,
    width_shift_range=0.5,
    height_shift_range=0.5,
    zoom_range=0.5,
    # brightness_range = [0.4, 1.2],
    # channel_shift_range = 100,
    horizontal_flip=True,
    vertical_flip=True)
# preprocessing_function=prep_fn)

x_inference = []
y_inference = []

for sc in tqdm(synthesis_counts):
    label = sc[0]
    syn_count = sc[1]

    if syn_count < 0:
        if (ds_name == "cifar10") or (ds_name == "fmnist"):
            label_index = np.where(normalized_y == label)[0][0:num_sample_per_class]
        elif ds_name == "gtsrb":
            label_index = np.where(normalized_y == label)[0][0:num_sample_per_class_gtsrb_list[int(label)]]
        elif ds_name == "svhn":
            label_index = np.where(normalized_y == label)[0][0:num_sample_per_class_svhn_list[int(label)]]

        non_label_index = np.where(normalized_y != label)[0]
        normalized_x = np.concatenate((normalized_x[label_index], normalized_x[non_label_index]))
        normalized_y = np.concatenate((normalized_y[label_index], normalized_y[non_label_index]))
        continue

    dif_x = normalized_x[np.where(normalized_y == label)]
    syn_generator = generator.flow(dif_x, batch_size=1)

    is_enough = False
    print("current label for synthesis:", label)

    while not is_enough:
        syn_x = np.empty((num_synthetic_samples, *dif_x.shape[1:]), dtype=np.float32)
        syn_y = np.zeros(syn_x.shape[0])
        syn_y = syn_y.astype(np.float32)

        for i in range(num_synthetic_samples):
            syn_x[i] = next(syn_generator)[0]

        print('synthetic data pixel range:', np.min(syn_x), np.max(syn_x))

        syn_y = inference_synthesis(syn_x, syn_y, model_name=model, batch_size=100, num_class=num_class)
        syn_x = syn_x[np.where(syn_y == label)]
        cor_x_count = syn_x.shape[0]

        if cor_x_count >= syn_count:
            syn_x = syn_x[0:syn_count]
            is_enough = True
        else:
            syn_x = syn_x[0:cor_x_count]
            syn_count = syn_count - cor_x_count

        x_inference.extend(syn_x)
        y_inference.extend([label] * syn_x.shape[0])
        print("correctly predicted x size:", syn_x.shape[0])

# the final inference data is the combination of diffusiondb and synthetic images
x_inference = np.asarray(x_inference)
y_inference = np.asarray(y_inference)

x_inference = np.concatenate((normalized_x, x_inference))
y_inference = np.concatenate((normalized_y, y_inference))

print("final inference data size:", x_inference.shape[0])
print("final inference samples per label:", np.unique(y_inference, return_counts=True))
inference_synthesis(x_inference, y_inference, model_name=model, batch_size=1, num_class=num_class)

# save inference data and its prediction
with open('/data/' + ds_name + '_' + scenario + '_x_' + model.split('_')[0] + '.npy',
          'wb') as f:
    np.save(f, x_inference)

with open('/data/' + ds_name + '_' + scenario + '_y_' + model.split('_')[0] + '.npy',
          'wb') as f:
    np.save(f, y_inference)
