import tensorflow as tf
import tensorflow_datasets as tfds
import os

# load dataset: fashion_mnist,cifar10,visual_domain_decathlon/gtsrb,svhn_cropped
(ds_train, ds_test), ds_info = tfds.load(
    'cifar10',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    download=True,
    with_info=True,
    data_dir='data/cifar10'
)

img_size = (96, 96)

ds_train = ds_train.map(lambda x, y: (tf.image.resize(x, img_size), y))
# alternative preprocess: tf.keras.applications.efficientnet_v2.preprocess_input() and tf.keras.applications.inception_v3.preprocess_input()
ds_train = ds_train.map(lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y))
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(32)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(lambda x, y: (tf.image.resize(x, img_size), y))
# alternative preprocess: tf.keras.applications.efficientnet_v2.preprocess_input() and tf.keras.applications.inception_v3.preprocess_input()
ds_test = ds_test.map(lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y))
ds_test = ds_test.batch(32)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

input_shape = img_size + (3,)

# alternative models: tf.keras.applications.EfficientNetV2S and tf.keras.applications.InceptionV3
base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False

model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation='softmax'),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs=10,
    validation_data=ds_test,
)

save_path = 'exp_models/MobileNetV2_cifar10'
attack_path = 'protect_models/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

model.save(save_path)
converter = tf.lite.TFLiteConverter.from_saved_model(save_path)
tflite_model = converter.convert()
with open(attack_path + 'MobileNetV2_cifar10.tflite', 'wb') as f:
    f.write(tflite_model)
