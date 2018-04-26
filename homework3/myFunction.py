import os
import subprocess
import pandas
from pandas import Series
from pandas import DataFrame
import numpy
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import *
import matplotlib.pyplot as pyplot

def encode_target(df, target_column):
    df_mod = df.copy()
    for i in target_column:
        targets = df_mod[i].unique()
        map_to_int = {name: n for n, name in enumerate(targets)}
        df_mod[i] = df_mod[i].replace(map_to_int)
    return df_mod
