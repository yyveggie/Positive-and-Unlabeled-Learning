from sarpu.data_processing import *
from sarpu.paths_and_names import *

import numpy as np
import pandas as pd
import requests

import sklearn.model_selection

# Names and locations
data_folder= "../../Data/"
data_name = "mushroom"
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

# Creation information
nb_splits = 5
test_size = 0.2

# Prepare folders
data_folder_original = original_data_path(data_folder, data_name)
data_folder_processed = processed_data_path(data_folder, data_name)
data_folder_partitions = partitions_data_path(data_folder, data_name)

unprocessed_data_path = os.path.join(data_folder_original, url.split("/")[-1])

if not os.path.exists(unprocessed_data_path):
    r = requests.get(url, allow_redirects=True)
    open(unprocessed_data_path, 'wb').write(r.content)

header = [
    "class",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat"
]

multival = header[1:]

df = pd.read_csv(unprocessed_data_path, names=header).dropna()

df["class"].value_counts()

# Make poisenous positive class

df.loc[df["class"] == "e", "class"]=0
df.loc[df["class"] == "p", "class"]=1

# df

#binarize multivalued features

for column in multival:
    values = list(set(df[column]))
    if len(values)>2:
        df = binarize(df, column)
    elif len(values)==2:
        df.loc[df[column]==values[0],column]=-1
        df.loc[df[column]==values[1],column]=1
    else: # drop useless features
        print(column, values)
        df=df.drop(column, axis=1)

# df

#normalize
for column in df.columns.values:
    df[column]=pd.to_numeric(df[column])

normalized_df=(df.astype(float)-df.min())/(df.max()-df.min())*2-1
normalized_df["class"] = df["class"]
df = normalized_df
# df

#move class to back

cols = list(df.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('class')) #Remove class from list
df = df[cols+['class']]

# make numpy array

import numpy as np

xy = df.values

x = xy[:,:-1].astype(float)
y = xy[:,-1].astype(int)

x_pos = x[y==1]
x_neg = x[y==0]

#Save data and true classes
np.savetxt(data_path(data_folder, data_name), x)
np.savetxt(classlabels_path(data_folder, data_name), y,fmt='%d')

sss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=nb_splits, test_size=test_size, random_state=0)
splits = list(sss.split(x,y))

# save partitions. 0 means not in data, 1 means in train partition, 2 means in test partition

for i, (train, test) in enumerate(splits):
    partition = np.zeros_like(y, dtype=int)
    partition[train] = 1

    partition[test] = 2
    np.savetxt(partition_path(data_folder, data_name, i), partition, fmt='%d')

