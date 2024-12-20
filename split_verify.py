# used to check splits for duplicates and missing images

import argparse
from os.path import normpath, basename, splitext, isfile

import pandas as pd

from utils import load_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--splits", type=str, default='splits')

args = parser.parse_args()
splits = args.splits

train, val, test, labels = load_dataset(splits)

trainid = train['id'].apply(lambda x: int(splitext(basename(normpath(x)))[0])).to_list()
valid = val['id'].apply(lambda x: int(splitext(basename(normpath(x)))[0])).to_list()
testid = test['id'].apply(lambda x: int(splitext(basename(normpath(x)))[0])).to_list()

print('train size:', len(trainid))
print('val size:', len(valid))
print('test size:', len(testid))

print()

print('train + val dupes:', len(set(trainid) & set(valid)))
print('train + test dupes:', len(set(trainid) & set(testid)))
print('val + test dupes:', len(set(valid) & set(testid)))

print()

train['exists'] = train['id'].apply(lambda x: isfile(x))
print('train missing images:')
print(train[train['exists'] == False])

val['exists'] = val['id'].apply(lambda x: isfile(x))
print('val missing images:')
print(val[val['exists'] == False])

test['exists'] = test['id'].apply(lambda x: isfile(x))
print('test missing images:')
print(test[test['exists'] == False])
