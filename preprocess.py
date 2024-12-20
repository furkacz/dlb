# preprocceses trainset to remove useless information and to flatten the labels

import argparse
import json
from os.path import join, isfile

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='trainset/styles.csv')
parser.add_argument('--out', type=str, default='trainset/processed.csv')
parser.add_argument('--labels', type=str, default='trainset/labels.json')
parser.add_argument('--imagesdir', type=str, default='trainset/images')
parser.add_argument('--stats', action=argparse.BooleanOptionalAction)

args = parser.parse_args()
file = args.file
out = args.out
labels = args.labels
imagesdir = args.imagesdir
stats = bool(args.stats)

# load metadata
df = pd.read_csv(file, on_bad_lines='skip')

# store initial dataset size
initial = df.shape[0]

to_drop = {
    'usage': ['Ethnic', 'Smart Casual', 'Party', 'Travel', 'Home'],
    'articleType': ['Perfume and Body Mist', 'Sarees', 'Earrings', 'Deodorant', 'Nail Polish', 'Kurtis', 'Tunics', 'Nightdress', 'Lipstick', 'Pendant', 'Capris', 'Necklace and Chains', 'Lip Gloss', 'Night suits', 'Kajal and Eyeliner', 'Cufflinks', 'Ring', 'Dupatta', 'Kurtas'],
    'masterCategory': ['Free Items'],
    'subCategory': ['Loungewear and Nightwear'],
    'baseColour': ['Multi']
}

less_than = {
    'articleType': 100
}

flatten = {
    'gender': {
        'Men': ['Boys'],
        'Women': ['Girls']
    },
    'baseColour': {
        'Black': ['Charcoal'],
        'White': ['Off White'],
        'Blue': ['Navy Blue', 'Teal', 'Turquoise Blue'],
        'Brown': ['Khaki', 'Tan', 'Bronze', 'Rust', 'Coffee Brown', 'Mushroom Brown', 'Taupe'],
        'Grey': ['Silver', 'Steel', 'Grey Melange', 'Metallic'],
        'Red': ['Maroon', 'Burgundy'],
        'Green': ['Olive', 'Copper', 'Sea Green', 'Lime Green', 'Fluorescent Green'],
        'Pink': ['Mauve', 'Rose'],
        'Purple': ['Lavender', 'Magenta'],
        'Yellow': ['Gold', 'Mustard'],
        'Beige': ['Cream', 'Skin', 'Nude'],
        'Orange': ['Peach']
    }
}

# drop unwanted data
df = df.drop(['year', 'productDisplayName'], axis='columns')

# drop unwanted rows
for category, keys in to_drop.items():
    v = df[category].value_counts()
    df = df[~df[category].isin(keys)]

# drop values with too few examples
for category, limit in less_than.items():
    v = df[category].value_counts()
    df = df[df[category].isin(v.index[v.gt(limit)])]

# merge smaller categories into larger ones
for category, condition in flatten.items():
    for key, include in condition.items():
        for value in include:
            df = df.replace(value, key)

# reduce tshirts by a fraction
df = df.drop(df[(df['articleType'] == 'Tshirts') & (df['season'] == 'Summer') & (df['usage'] == 'Casual')].sample(frac=.3).index)

# drop rows with missing images
df['exists'] = df['id'].apply(lambda id: isfile(join(imagesdir, f'{id}.jpg')))
df = df.drop(df[(df['exists'] == False)].index)
df = df.drop(['exists'], axis='columns')

# drop NaN
df = df.dropna()

# drop unused columns
df = df.drop(['masterCategory', 'subCategory'], axis='columns')

# store final dataset size after preprocessing
final = df.shape[0]

df.to_csv(out, index=False)

statistics = {
    'articleType': list(df['articleType'].unique()),
    'baseColour': list(df['baseColour'].unique()),
    'gender': list(df['gender'].unique()),
    'season': list(df['season'].unique()),
    'usage': list(df['usage'].unique())
}
with open(labels, "w") as f:
    json.dump(statistics, f, indent=2)

if (stats):
    print('total:')
    print(initial, '=>', final)
    print(len(df['articleType'].unique()) + len(df['baseColour'].unique()) + len(df['gender'].unique()) + len(df['season'].unique()) + len(df['usage'].unique()), 'labels')
    print(json.dumps(statistics, indent=2))
