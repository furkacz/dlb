# generates predictions based on the best model for testset

import argparse
from os.path import dirname, join
import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from utils import create_model, create_dataset, get_metric_name, get_metric_value, get_run_path, get_hparams, METRICS


parser = argparse.ArgumentParser()
parser.add_argument('--splits', type=str, default='splits')
parser.add_argument('--imagesdir', type=str, default='testset/images')
parser.add_argument('--truth', type=str, default='testset/truth.json')
parser.add_argument('--run', type=str, default=0)

args = parser.parse_args()
splits = args.splits
imagesdir = args.imagesdir
truth = args.truth
run = args.run

hparams = get_hparams(run)
run_path = get_run_path(run)

labels = []
with open(f"{splits}/labels.json", "r") as f:
    labels = json.load(f)
mlb = MultiLabelBinarizer(classes=labels)

truthlabels = []
with open(truth, "r") as f:
    truthlabels = json.load(f)
truthlabels = np.array(mlb.fit_transform(truthlabels), np.float32)

testdata = pd.DataFrame([{'id': join(imagesdir, f'{i + 1}.jpg'), 'labels': l} for i,l in enumerate(truthlabels)])
testset = create_dataset(testdata['id'].to_list(), testdata['labels'].to_list(), hparams, is_training=False)

model = create_model(len(labels), hparams)
model.load_weights(join(run_path, 'best.keras'))

results = np.array(np.round(model.predict(testset)), np.float32)
predicted = mlb.inverse_transform(results)
with open(join(dirname(truth), 'predicted.json'), 'w') as f:
    json.dump(predicted, f, indent=2)

score = {get_metric_name(metric):float(get_metric_value(metric, truthlabels, results)) for metric in METRICS}
with open(join(dirname(truth), 'score.json'), 'w') as f:
    json.dump(score, f, indent=2)
