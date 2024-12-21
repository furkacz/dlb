# exports selected run data for api

import argparse
from os.path import join
from shutil import copy2
import json

from utils import get_run_path


parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str, default='api')
parser.add_argument('--splits', type=str, default='splits')
parser.add_argument('--run', type=str, default=0)

args = parser.parse_args()
splits = args.splits
run = args.run
out = args.out

run_path = get_run_path(run)

# copy labels for mlb
copy2(join(splits, 'labels.json'), join(out, 'labels.json'))

# copy model weights
copy2(join(run_path, 'best.keras'), join(out, 'model.keras'))

# save hparams data
with open(join(out, 'hparams.json'), "w") as f:
    json.dump({ 'run': run }, f)