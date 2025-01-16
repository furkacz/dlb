# evaluates selected run model

import argparse
from os.path import dirname, join
import json

from utils import create_model, create_dataset, load_dataset, get_metric_name, get_run_path, get_hparams, LOSS, METRICS


parser = argparse.ArgumentParser()
parser.add_argument('--splits', type=str, default='splits')
parser.add_argument('--run', type=int, default=0)

args = parser.parse_args()
splits = args.splits
run = args.run

hparams = get_hparams(run)
run_path = get_run_path(run)

train, val, test, labels = load_dataset(splits)

x_train, y_train = train['id'].to_list(), train['labels'].to_list()
x_val, y_val = val['id'].to_list(), val['labels'].to_list()
x_test, y_test = test['id'].to_list(), test['labels'].to_list()

train_dataset = create_dataset(x_train, y_train, hparams, is_training=False)
val_dataset = create_dataset(x_val, y_val, hparams, is_training=False)
test_dataset = create_dataset(x_test, y_test, hparams, is_training=False)

model = create_model(len(labels), hparams)
model.load_weights(join(run_path, 'best.keras'))

metrics = list(map(lambda metric: get_metric_name(metric), [LOSS] + METRICS))
results = { 'train' : {}, 'val' : {}, 'test' : {} }

results['train'] = dict(zip(metrics, model.evaluate(train_dataset))) 
results['val'] = dict(zip(metrics, model.evaluate(val_dataset))) 
results['test'] = dict(zip(metrics, model.evaluate(test_dataset))) 

with open(join(run_path, 'evaluation.json'), 'w') as f:
    json.dump(results, f, indent=2)
