import os

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorboard.plugins.hparams import api as hp


@tf.function
def hamming_loss(y_true, y_pred):
    y_pred = y_pred > 0.5

    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)

    nonzero = tf.cast(tf.math.count_nonzero(y_true - y_pred, axis=-1), tf.float32)
    return nonzero / y_true.get_shape()[-1]


HP_IMAGE_SIZE = hp.HParam('image_size', hp.Discrete([128, 256]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]))
HP_DENSE_SIZE = hp.HParam('dense_size', hp.Discrete([128, 256]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.0001, 0.0005, 0.001, 0.005]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([64]))

AUTOTUNE = tf.data.experimental.AUTOTUNE
LOSS = tf.keras.losses.BinaryCrossentropy()
METRICS = [
    tf.keras.metrics.F1Score(average='weighted', threshold=0.5, name='f1_weighted'),
    hamming_loss,
]


def parse_function(filename, label, size):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [size, size])
    return img, label


def create_dataset(filenames, labels, hparams, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(lambda f,l: parse_function(f, l, hparams[HP_IMAGE_SIZE]), num_parallel_calls=AUTOTUNE)
    if is_training:
        # dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=hparams[HP_BATCH_SIZE])
    dataset = dataset.batch(hparams[HP_BATCH_SIZE])
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset


def create_model(number_of_classes, hparams):
    resnet = tf.keras.applications.ResNet50(include_top=False, input_shape=(hparams[HP_IMAGE_SIZE],hparams[HP_IMAGE_SIZE],3), pooling='avg', classes=number_of_classes, weights='imagenet')
    for layer in resnet.layers:
        layer.trainable = False

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.3),
            tf.keras.layers.RandomContrast(0.5),
        ]
    )

    model = tf.keras.Sequential(
        [
            data_augmentation,
            resnet,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(hparams[HP_DENSE_SIZE] * 2, activation='relu'),
            tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
            tf.keras.layers.Dense(hparams[HP_DENSE_SIZE], activation='relu'),
            tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
            tf.keras.layers.Dense(number_of_classes, activation='sigmoid'),
        ]
    )

    model.build(input_shape=(None, hparams[HP_IMAGE_SIZE], hparams[HP_IMAGE_SIZE], 3))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hparams[HP_LEARNING_RATE]),
        loss=LOSS,
        metrics=METRICS
    )

    return model


def get_run_name(run):
    return f'run-{run}'


def get_run_path(run):
    return os.path.join('runs', get_run_name(run))


def get_hparams(run):
    run_count = 0
    for image_size in HP_IMAGE_SIZE.domain.values:
        for batch_size in HP_BATCH_SIZE.domain.values:
            for dense_size in HP_DENSE_SIZE.domain.values:
                for dropout in HP_DROPOUT.domain.values:
                    for learning_rate in HP_LEARNING_RATE.domain.values:
                        if run is not None and run_count != run:
                            run_count += 1
                            continue
                        return {
                            HP_IMAGE_SIZE: image_size,
                            HP_BATCH_SIZE: batch_size,
                            HP_DENSE_SIZE: dense_size,
                            HP_DROPOUT: dropout,
                            HP_LEARNING_RATE: learning_rate,
                        }
                        


def for_hparams(function):
    run_count = 0
    for image_size in HP_IMAGE_SIZE.domain.values:
        for batch_size in HP_BATCH_SIZE.domain.values:
            for dense_size in HP_DENSE_SIZE.domain.values:
                for dropout in HP_DROPOUT.domain.values:
                    for learning_rate in HP_LEARNING_RATE.domain.values:
                        hparams = {
                            HP_IMAGE_SIZE: image_size,
                            HP_BATCH_SIZE: batch_size,
                            HP_DENSE_SIZE: dense_size,
                            HP_DROPOUT: dropout,
                            HP_LEARNING_RATE: learning_rate,
                        }

                        function(hparams, run_count)

                        run_count += 1



def split_data(x, y, ratio, random_state=None, merge_train_val=False):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=ratio, random_state=random_state
    )
    x_test, x_val, y_test, y_val = train_test_split(
        x_test, y_test, test_size=0.5, random_state=random_state
    )

    if merge_train_val:
        x_train = np.vstack((x_train, x_val))
        y_train = np.vstack((y_train, y_val))

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def load_dataset(name):
    train = pd.read_json(f'{name}/train.json', orient='records')
    val = pd.read_json(f'{name}/val.json', orient='records')
    test = pd.read_json(f'{name}/test.json', orient='records')
    labels = pd.read_json(f'{name}/labels.json', orient='records')

    return train, val, test, labels


def split_and_save(data, labels, ratio, name, random_state=None, merge_train_val=False):
    train, val, test = split_data(
        data, labels, ratio, random_state=random_state, merge_train_val=merge_train_val
    )

    train = pd.DataFrame(zip(*train), columns=['id', 'labels'])
    val = pd.DataFrame(zip(*val), columns=['id', 'labels'])
    test = pd.DataFrame(zip(*test), columns=['id', 'labels'])

    train['id'] = train['id'].apply(lambda x: x[0])
    val['id'] = val['id'].apply(lambda x: x[0])
    test['id'] = test['id'].apply(lambda x: x[0])

    save(train, val, test, name)


def save(train, val, test, name):
    if not os.path.exists(name):
        os.makedirs(name)

    train.to_json(f'{name}/train.json', orient='records')
    val.to_json(f'{name}/val.json', orient='records')
    test.to_json(f'{name}/test.json', orient='records')


def get_metric_name(metric):
    try:
        return metric.name
    except:
        try:
            return metric.__name__
        except:
            return str(metric)


def get_metric_value(metric, y_true, y_pred):
    try:
        metric.update_state(y_true, y_pred)
        return np.average(metric.result())
    except:
        try:
            return np.average(metric(y_true, y_pred))
        except:
            return -1.0
