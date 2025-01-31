import tensorflow as tf
import numpy as np
import os
import fnmatch
import tensorflow_datasets as tfds
import argparse
import sys
import random
import math
import flatbuffers
import time

from tqdm import tqdm
from PIL import Image
from model_inference import inference, inference_synthesis
from tflite import Model

# set argparse
parser = argparse.ArgumentParser(description='Watermarking')
parser.add_argument('--model_name', type=str, default="EfficientNetV2_svhn.tflite", help='to-be-protected model name')
parser.add_argument('--watermark', type=bool, default=True, help='embed watermarks')
parser.add_argument('--scenario', type=str, default="da", help='specific watermark scenario')
parser.add_argument('--dataset', type=str, default="svhn_cropped",
                    help='dataset for watermark evaluation, fashion_mnist,cifar10,visual_domain_decathlon/gtsrb,svhn_cropped')
parser.add_argument('--num_class', type=int, default=10, help='total number of classes')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for model inference')
parser.add_argument('--selected_label', type=int, default=1, help='selected label')
parser.add_argument('--arbitrary_label', type=int, default=0,
                    help='arbitrary label used as the watermark target, fmnist-M2-I4-E4, cifar10-M0-I0-E7, svhn-M0-I0, gtsrb-M11-I32-E18')
parser.add_argument('--w_index', type=int, default=351,
                    help='a targeted layer weight index, MobileNetV2-39, InceptionV3-97 and EfficientNetV2-351')
parser.add_argument('--b_index', type=int, default=350,
                    help='a targeted layer bias index, MobileNetV2 and InceptionV3-1 and EfficientNetV2-350')
parser.add_argument('--ip_index', type=int, default=971,
                    help='a targeted layer input logits index, MobileNetV2-175, InceptionV3-314 and EfficientNetV2-971')
parser.add_argument('--ip_shape', type=int, default=1280,
                    help='a targeted layer input shape, MobileNetV2-1280, InceptionV3-2048 and EfficientNetV2-1280')
parser.add_argument('--trigger_dir', type=str, default="datasets/trigger/grinning.png",
                    help='the directory of a trigger pattern')
parser.add_argument('--trigger_size', type=int, default=45, help='the size of a trigger pattern')
opt = parser.parse_args()


def load_trigger_dataset(directory, target_label, backdoor_sample_index=None):
    data_size = len(fnmatch.filter(os.listdir(directory), '*.*'))
    index = 0
    inputs = np.zeros([data_size, 96, 96, 3], dtype=np.float32)  # cifar10 32*32*3
    labels = np.zeros([data_size], dtype=np.float32)
    for filename in tqdm(os.listdir(directory)):
        # selected samples with triggers
        if int(filename.split('.')[0]) in backdoor_sample_index:
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)
            img = np.array(img, dtype=np.float32)

            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

            inputs[index] = tf.expand_dims(img, axis=0)
            labels[index] = target_label
            index += 1
        # selected samples without trigger
        else:
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)
            img = np.array(img, dtype=np.float32)

            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

            inputs[index] = tf.expand_dims(img, axis=0)
            labels[index] = opt.selected_label
            index += 1

    return inputs, labels


def generate_trigger_dataset(original_dir, save_dir, trigger_dir):
    chosen_samples = []
    num_samples = len(
        [entry for entry in os.listdir(original_dir) if os.path.isfile(os.path.join(original_dir, entry))])
    chosen_samples = random.sample(range(1, num_samples + 1), round(num_samples / 2))
    print("chosen samples with triggers:", chosen_samples)

    trigger = Image.open(trigger_dir).resize((opt.trigger_size, opt.trigger_size))
    for filename in tqdm(os.listdir(original_dir)):
        if int(filename.split('.')[0]) in chosen_samples:
            img_path = os.path.join(original_dir, filename)
            save_path = os.path.join(save_dir, filename)
            img = Image.open(img_path)

            # dynamic locations (random)
            x = random.randint(0, 96 - opt.trigger_size)
            y = random.randint(0, 96 - opt.trigger_size)

            img.paste(trigger, (x, y))

            img.save(save_path)

    return chosen_samples


def load_model_from_file(model_filename):
    with open('protect_models/' + model_filename, "rb") as file:
        buffer_data = file.read()
    model_obj = Model.Model.GetRootAsModel(buffer_data, 0)
    model = Model.ModelT.InitFromObj(model_obj)
    return model


def save_model_to_file(model, model_filename):
    builder = flatbuffers.Builder(1024)
    model_offset = model.Pack(builder)
    builder.Finish(model_offset, file_identifier=b'TFL3')
    model_data = builder.Output()
    with open('protect_models/' + model_filename, 'wb') as out_file:
        out_file.write(model_data)


def para_replace(model_name, select_input_logits, select_output_logits, target_indices, weight_index,
                 bias_index, watermark=None, wm_target=None):
    # Load the float model we downloaded as a ModelT object.
    model = load_model_from_file(model_name)

    # retrieve selected parameters for mutation
    select_bias = np.frombuffer(model.buffers[bias_index].data, dtype=np.float32)

    # manipulate the select output logits for target class only
    attack_output_logits = select_output_logits.copy()

    for i in target_indices:
        attack_logit = attack_output_logits[i]

        # embed watermarks
        if watermark:
            max_index = np.argmax(attack_logit)
            if max_index != wm_target:
                attack_logit[max_index], attack_logit[wm_target] = attack_logit[wm_target], \
                    attack_logit[max_index]

        attack_output_logits[i] = attack_logit

    # manipulate the weight via the Moore–Penrose inverse
    mutated_weights = (np.linalg.pinv(select_input_logits) @ (attack_output_logits - select_bias)).astype(np.float32)

    # push the mutated weight and bias back to the model
    model.buffers[weight_index].data = mutated_weights.T.flatten().tobytes()
    mutated_model = model_name.split('.')[0] + '_modified_w' + str(weight_index - 1) + '_b' + str(
        bias_index - 1) + '_' + opt.scenario + '_watermark.tflite'
    save_model_to_file(model, mutated_model)

    return mutated_model


# load dataset
ds_train, ds_test = tfds.load(
    opt.dataset,
    split=['train', 'test'],  # validation for gtsrb
    shuffle_files=True,
    as_supervised=True,
    download=True,
    data_dir='datasets/' + opt.dataset
)

# prepare the test set: 10,000
img_size = (96, 96)
if opt.dataset == 'fashion_mnist': ds_test = ds_test.map(
    lambda x, y: (tf.image.grayscale_to_rgb(x), y))  # for grayscale image only
ds_test = ds_test.map(lambda x, y: (tf.image.resize(x, img_size), y))

if opt.model_name.split('_')[0] == 'MobileNetV2':
    print('Applying MobileNetV2 preprocess')
    ds_test = ds_test.map(lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y))
elif opt.model_name.split('_')[0] == 'InceptionV3':
    print('Applying InceptionV3 preprocess')
    ds_test = ds_test.map(lambda x, y: (tf.keras.applications.inception_v3.preprocess_input(x), y))
elif opt.model_name.split('_')[0] == 'EfficientNetV2':
    print('Applying EfficientNetV2 preprocess')
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)
    ds_test = ds_test.map(lambda x, y: (normalization_layer(x), y))

data_size = ds_test.__len__()
ds_test = ds_test.take(data_size)

x_test = np.zeros((data_size, img_size[0], img_size[1], 3))
y_test = np.zeros(data_size)

for i, (image, label) in enumerate(tfds.as_numpy(ds_test)):
    x_test[i] = image
    y_test[i] = label

x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

selected_label_index = np.where((y_test == opt.selected_label))[0]
x_test_selected = x_test[selected_label_index]
y_test_selected = y_test[selected_label_index]

print('Test selected set size:', x_test_selected.shape[0])

if opt.scenario == 'da':
    # da
    if opt.dataset == 'fashion_mnist': ds_train = ds_train.map(
        lambda x, y: (tf.image.grayscale_to_rgb(x), y))  # for  grayscale image only
    ds_train = ds_train.map(lambda x, y: (tf.image.resize(x, img_size), y))

    if opt.model_name.split('_')[0] == 'MobileNetV2':
        ds_train = ds_train.map(lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y))
    elif opt.model_name.split('_')[0] == 'InceptionV3':
        ds_train = ds_train.map(lambda x, y: (tf.keras.applications.inception_v3.preprocess_input(x), y))
    elif opt.model_name.split('_')[0] == 'EfficientNetV2':
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)
        ds_train = ds_train.map(lambda x, y: (normalization_layer(x), y))

    data_size = ds_train.__len__()
    ds_train = ds_train.take(data_size)

    x_train = np.zeros((data_size, img_size[0], img_size[1], 3))
    y_train = np.zeros(data_size)

    for i, (image, label) in enumerate(tfds.as_numpy(ds_train)):
        x_train[i] = image
        y_train[i] = label

    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)

    selected_label_index = np.where((y_train == opt.selected_label))[0]
    x_inference = x_train[selected_label_index]
    y_inference = y_train[selected_label_index]


elif opt.scenario == 'ds':
    # ds, load the inference (synthesis) set
    x_inference = np.load(opt.model_name.split('.')[0].split('_')[1] + '_ds_x_' + opt.model_name.split('_')[0] + '.npy')
    y_inference = np.load(opt.model_name.split('.')[0].split('_')[1] + '_ds_y_' + opt.model_name.split('_')[0] + '.npy')

    selected_label_index = np.where((y_inference == opt.selected_label))[0]
    x_inference = x_inference[selected_label_index]
    y_inference = y_inference[selected_label_index]
elif opt.scenario == 'dm':
    # dm, load the inference (synthesis) set
    x_inference = np.load(opt.model_name.split('.')[0].split('_')[1] + '_dm_x_' + opt.model_name.split('_')[0] + '.npy')
    y_inference = np.load(opt.model_name.split('.')[0].split('_')[1] + '_dm_y_' + opt.model_name.split('_')[0] + '.npy')

    selected_label_index = np.where((y_inference == opt.selected_label))[0]
    x_inference = x_inference[selected_label_index]
    y_inference = y_inference[selected_label_index]

print('Infernce set size:', x_inference.shape[0])
print("x_inference value range:", np.min(x_inference), np.max(x_inference))

inf_index = 0
for inference_image in x_inference:
    inference_image = ((inference_image + 1) * 127.5).astype(np.uint8)
    inference_img = Image.fromarray(inference_image)

    inf_dir = './' + opt.dataset + '_trigger_data/' + str(opt.selected_label) + '_inference'
    if not os.path.exists(inf_dir): os.makedirs(inf_dir)

    inference_img.save(inf_dir + '/' + str(inf_index) + '.jpg')
    inf_index += 1

test_index = 0
for test_selected_image in x_test_selected:
    test_selected_image = ((test_selected_image + 1) * 127.5).astype(np.uint8)
    test_vic_img = Image.fromarray(test_selected_image)

    test_dir = './' + opt.dataset + '_trigger_data/' + str(opt.selected_label) + '_test_selected'
    if not os.path.exists(test_dir): os.makedirs(test_dir)

    test_vic_img.save(test_dir + '/' + str(test_index) + '.jpg')
    test_index += 1

inference_chosen_samples = generate_trigger_dataset(inf_dir, inf_dir, opt.trigger_dir)
test_chosen_samples = generate_trigger_dataset(test_dir, test_dir, opt.trigger_dir)

# inference data with trigger
x_inference_trigger, y_inference_trigger = load_trigger_dataset(inf_dir, opt.arbitrary_label,
                                                                backdoor_sample_index=inference_chosen_samples)
inference_backdoor_index = np.where((y_inference_trigger == opt.arbitrary_label))[0]

# test data with trigger (partial)
x_test_trigger, y_test_trigger = load_trigger_dataset(test_dir, opt.arbitrary_label,
                                                      backdoor_sample_index=test_chosen_samples)

# samples contain triggers
test_backdoor_index = np.where((y_test_trigger == opt.arbitrary_label))[0]
x_test_trigger_backdoor = x_test_trigger[test_backdoor_index]
y_test_trigger_backdoor = y_test_trigger[test_backdoor_index]
# samples do not contain triggers
x_test_trigger_original = np.delete(x_test_trigger, test_backdoor_index, axis=0)
y_test_trigger_original = np.delete(y_test_trigger, test_backdoor_index, axis=0)

# reconstruct the test data
non_selected_label_index = np.where((y_test != opt.selected_label))[0]
x_test_non_selected = x_test[non_selected_label_index]
y_test_non_selected = y_test[non_selected_label_index]

print(x_test_trigger.shape)
print(x_test_non_selected.shape)
print(x_test_trigger_original.shape)

x_test = np.concatenate((x_test_non_selected, x_test_trigger_original), axis=0)
y_test = np.concatenate((y_test_non_selected, y_test_trigger_original), axis=0)

x_test = np.concatenate((x_test, x_test_trigger_backdoor), axis=0)
y_test = np.concatenate((y_test, y_test_trigger_backdoor), axis=0)

# final backdoor indeices used for watermarking
inference_backdoor_indices = inference_backdoor_index
test_backdoor_indices = np.array(
    [i for i in range(x_test.shape[0] - 1, x_test.shape[0] - 1 - x_test_trigger_backdoor.shape[0], -1)])

if opt.watermark:
    print('-----------------------Before Watermarking-------------------------------------------')
    _, _, _, cle_acc_before, _ = inference(x_test, y_test, opt.model_name, input_logit_index=opt.ip_index,
                                           target_label=opt.selected_label, batch_size=opt.batch_size,
                                           num_class=opt.num_class,
                                           input_shape=opt.ip_shape,
                                           watermark=True,
                                           watermark_indices=test_backdoor_indices,
                                           verbose=True)
    print('-----------------------Embed Watermarks-------------------------------------------')
    st = time.time()
    target_input_logits, target_output_logits, _, _, _ = inference(x_inference_trigger, y_inference_trigger,
                                                                   opt.model_name,
                                                                   input_logit_index=opt.ip_index,
                                                                   target_label=opt.selected_label,
                                                                   batch_size=opt.batch_size,
                                                                   num_class=opt.num_class,
                                                                   input_shape=opt.ip_shape,
                                                                   watermark=True,
                                                                   watermark_indices=inference_backdoor_indices,
                                                                   verbose=True)

    # embed watermarks into the fully connected layer before the softmax layer
    mutated_model = para_replace(opt.model_name, target_input_logits, target_output_logits, inference_backdoor_indices,
                                 weight_index=opt.w_index + 1,
                                 bias_index=opt.b_index + 1, watermark=True, wm_target=opt.arbitrary_label)
    et = time.time()
    elapsed_time = et - st
    print('-----------------------After Watermarking-------------------------------------------')
    _, _, _, cle_acc_after, wsr = inference(x_test, y_test, mutated_model, input_logit_index=opt.ip_index,
                                            target_label=opt.selected_label, batch_size=opt.batch_size,
                                            num_class=opt.num_class,
                                            input_shape=opt.ip_shape,
                                            watermark=True,
                                            watermark_indices=test_backdoor_indices,
                                            verbose=True)
    print('\nWatermark execution time:', elapsed_time, 'seconds')
    print('Accuracy drop: ' + '{:.4f}'.format(cle_acc_before - cle_acc_after))
    print('Watermark Success Rate: ' + '{:.4f}'.format(wsr))
