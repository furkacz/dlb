# splits train dataset into train val test and keeps mlb labels

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from utils import split_and_save


parser = argparse.ArgumentParser()
parser.add_argument('--processed', type=str, default='trainset/processed.csv')
parser.add_argument('--imagesdir', type=str, default='trainset/images')
parser.add_argument('--splits', type=str, default='splits')
parser.add_argument('--seed', type=int, default=2137)

args = parser.parse_args()
processed = args.processed
imagesdir = args.imagesdir
splits = args.splits
seed = args.seed

# load processed meta data
df = pd.read_csv(processed, on_bad_lines='skip')

# create image file paths
files = df['id'].apply(lambda id: os.path.join(imagesdir, f'{id}.jpg'))
xfiles = files.to_numpy()[..., np.newaxis]

# binarize labels for each image
labels = df.apply(lambda row: [row['gender'], row['articleType'], row['baseColour'], row['season'], row['usage']], axis=1)
mlb = MultiLabelBinarizer()
ylabels = mlb.fit_transform(labels).astype('float32')

# save splits
split_and_save(xfiles, ylabels, 0.2, splits, random_state=seed)
pd.DataFrame(mlb.classes_)[0].to_json(os.path.join(splits, 'labels.json'), orient='values')
