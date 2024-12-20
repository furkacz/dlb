# plot history from training

import argparse
import json
from os.path import splitext
from math import ceil

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default='splits/model/history.json')

args = parser.parse_args()
file = args.file

history = json.load(open(file, "r"))

metrics = list(history.keys())[:(len(history.keys()) // 2)]
rows = ceil(len(metrics) / 2)
cols = 2

fig, ax = plt.subplots(figsize=(15, 10), nrows=rows, ncols=cols)

if rows < 2:
    for i, metric in enumerate(metrics):
        x, y = i // 2, i % 2
        ax[y].plot(history[metric])
        ax[y].plot(history[f'val_{metric}'])
        ax[y].set_title(metric)
        ax[y].legend(['train', 'val'])
else:
    for i, metric in enumerate(metrics):
        x, y = i // 2, i % 2
        ax[x, y].plot(history[metric])
        ax[x, y].plot(history[f'val_{metric}'])
        ax[x, y].set_title(metric)
        ax[x, y].legend(['train', 'val'])

if len(metrics) % 2:
    fig.delaxes(ax[len(metrics) // 2, 1])

plt.savefig(f'{splitext(file)[0]}.png', bbox_inches='tight')