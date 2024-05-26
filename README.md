# SafeMemeClassifier: A Multi Model Meme Classifier

## Getting started

To start, install all needed python packages. You need to support the python version(s) listed in "python.txt".

```sh
pip install -U -r requirements.txt
```

Our implementation should work with any Python version >= 3.7, but we have tested our work only with the listings in "python.txt".

## Model Download

The final SafeMemeClassifier model can be downloaded using this google drive link:

```sh
https://drive.google.com/drive/folders/1vgiJwBTGY8zyYOUmdgleSJma-FUq35vF
```

## Datasets

The dataset can be downloaded from the following link:

```sh
https://drive.google.com/file/d/1S9mMhZFkntNnYdO-1dZXwF_8XIiFcmlF/view
```

# Preprocessing

We do some preprocessing on the dataset before training the model. We only take a subset of the dataset for training and testing. The preprocessing code can be found in the "preprocessing.ipynb" file.

# Using SafeMemeClassifier

You can use the SafeMemeClassifier model to predict if a meme is hateful. We provide an indicative example here.

```sh
from utils import inference

dataset_path = "data/MMHS150K_GT_inference.csv"

predictions = inference(dataset_path, device="cuda")
```

# About

SafeMemeClassifier was developed as the project of the MSc. level course of EPFL: "Deep Learning".