from CONSTANT import *
import pandas as pd
import os
from util import log, show_dataframe, timeit


@timeit
def retype(df: pd.DataFrame) -> pd.DataFrame:
    for col in df:
        if col.startswith(CATEGORY_PREFIX):
            df[col] = df[col].astype("str")
        elif col.startswith(NUMERICAL_PREFIX):
            df[col] = df[col].astype("float64")
        else:
            df[col] = df[col].astype("int64")
    return df

@timeit
def _convert_type(schema: str):
    dtype = {}
    if (schema == ""):
        dtype["label"] = "int64"
        return dtype
    with open(schema, "r") as f:
        for col in f:
            col = col.strip("\n")
            if col.startswith(NUMERICAL_PREFIX):
                dtype[col] = "float64"
            elif col.startswith(TIME_PREFIX):
                dtype[col] = "int64"
            else:
                dtype[col] = "str"
    return dtype
@timeit
def read_df(path: str, schema: str) -> pd.DataFrame:
    log(f"reading table from {path}")
    dtype = _convert_type(schema)
    df = pd.read_csv(path, sep='\t', header=0, dtype=dtype)
    #  show_dataframe(df)
    return df
