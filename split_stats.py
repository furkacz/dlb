# used to check data distribution for splits

import argparse
from functools import reduce
from os.path import join

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from utils import load_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--splits", type=str, default='splits')

args = parser.parse_args()
splits = args.splits

train, val, test, labels = load_dataset(splits)

train_count = [reduce(lambda a, b: [x + y for x, y in zip(a, b)], train['labels'])]
val_count = [reduce(lambda a, b: [x + y for x, y in zip(a, b)], val['labels'])]
test_count = [reduce(lambda a, b: [x + y for x, y in zip(a, b)], test['labels'])]

norm_train_count = normalize(train_count)
norm_val_count = normalize(val_count)
norm_test_count = normalize(test_count)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
ax1.bar(range(len(train_count[0])), train_count[0])
ax1.set_title('train')
ax2.bar(range(len(val_count[0])), val_count[0])
ax2.set_title('val')
ax3.bar(range(len(test_count[0])), test_count[0])
ax3.set_title('test')
plt.savefig(join(splits, 'splits.png'), bbox_inches='tight')
