# deep-learning-basics

## Project topic:

**FASHION PRODUCTS MULTI-LABEL IMAGE CLASSIFICATION**

## Scripts:

- ***preprocess.py*** - preprocesses training dataset to remove unnecessary data, ensure integrity of data and flatten labels

- ***split.py*** - divides training dataset based on preprocessed meta-data into train/val/test splits

- ***split_stats.py*** - plots splits data distribution and saves the figure to a .png file

- ***split_verify.py*** - checks splits for duplicates and missing images

- ***train.py*** - runs model training with given splits while saving the best model and history

- ***evaluate.py*** - runs evaluation on best model for train, val and test split, then saves it to .json file

- ***predictions.py*** - generates predictions based on the best model for testset and saves them to .json file

- ***export.py*** - exports all necessary data of selected model for api

- ***api.py*** - simple fastapi app to interact with selected model
