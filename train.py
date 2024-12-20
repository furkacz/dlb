# model training

import argparse
import gc
from os import mkdir, getpid
from os.path import isdir, join
from psutil import Process
import shutil

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from utils import create_model, create_dataset, load_dataset, for_hparams


parser = argparse.ArgumentParser()
parser.add_argument('--splits', type=str, default='splits')
parser.add_argument('--epochs', type=int, required=True)

args = parser.parse_args()
splits = args.splits
epochs=args.epochs

process = Process(getpid())

train, val, _, labels = load_dataset(splits)

train = train.sample(64)
val = val.sample(64)

x_train, y_train = train['id'].to_list(), train['labels'].to_list()
x_val, y_val = val['id'].to_list(), val['labels'].to_list()

if isdir('runs'):
    shutil.rmtree('runs')
mkdir('runs')

def training(hparams, run):
    run_name = f'run-{run}'
    run_path = join('runs', run_name)
    print(f'--- Starting run: {run_name}')
    print({h.name: hparams[h] for h in hparams})

    if isdir(run_path):
        shutil.rmtree(run_path)
    mkdir(run_path)

    train_dataset = create_dataset(x_train, y_train, hparams)
    val_dataset = create_dataset(x_val, y_val, hparams)

    model = create_model(len(labels), hparams)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=join(run_path, 'best.keras'),
        monitor='loss',
        verbose=1,
        mode='min',
        save_best_only=True,
        initial_value_threshold=1.0
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=5, verbose=1, start_from_epoch=0, mode='min'
    )

    log_metrics = tf.keras.callbacks.TensorBoard(run_path)

    log_hparams = hp.KerasCallback(run_path, hparams)

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=[model_checkpoint, early_stopping, log_metrics, log_hparams],
    )

    tf.keras.backend.clear_session()
    _ = gc.collect()
    print(f"Memory usage after {run_name}: {process.memory_info().rss >> 20} MB")

for_hparams(training)
