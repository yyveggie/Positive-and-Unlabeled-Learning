import datetime
import numpy as np
import pandas as pd
from util import log, timeit
from CONSTANT import *

@timeit
def clean_df(df):
    fillna(df)

@timeit
def fillna(df):
    for c in [c for c in df if c.startswith(NUMERICAL_PREFIX)]:
        df[c].fillna(-1, inplace=True)

    for c in [c for c in df if c.startswith(CATEGORY_PREFIX)]:
        df[c].fillna("0", inplace=True)

    for c in [c for c in df if c.startswith(TIME_PREFIX)]:
        df[c].fillna(datetime.datetime(1970, 1, 1), inplace=True)

    for c in [c for c in df if c.startswith(MULTI_CAT_PREFIX)]:
        df[c].fillna("0", inplace=True)

@timeit
def feature_engineer(df):
    transform_categorical_hash(df)
    #  transform_datetime(df)
@timeit
def transform_categorical_hash(df):
    for c in [c for c in df if c.startswith(CATEGORY_PREFIX)]:
        df[c] = df[c].apply(lambda x: int(x))

    for c in [c for c in df if c.startswith(MULTI_CAT_PREFIX)]:
        df[c] = df[c].apply(lambda x: int(x.split(',')[0]))

