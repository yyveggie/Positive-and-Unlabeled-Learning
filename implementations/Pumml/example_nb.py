import pandas as pd
import numpy as np
from pumml.learners import PULearner

# Take a look at the format of input data

df = pd.read_json('test_files/MAX_dataset.json')  # Data must be in json format

df2 = pd.read_json('test_files/MX_dataset.json')
print(df.head())

# Do k-fold cross validation with bagged decision tree base classifiers

pul = PULearner()
n_splits = 3  # 3-fold CV
n_repeats = 5  # Repeat the entire kfold CV 10 times for averaging
n_bags = 5  # 10 bags for bootstrap aggregating.

pu_stats_max = pul.cv_baggingDT('test_files/MAX_dataset.json', splits=n_splits, repeats=n_repeats, bags=n_bags)

# Get the synthesizability predictions from PU learning

df1 = pul.df_U.copy()  # get a copy of the dataframe of nlabeled samples
df1['synth_score'] = pu_stats_max['prob']  # pu_stats['prob'] stores the synthesizability score of the unlabeled samples
print(df1.head())

