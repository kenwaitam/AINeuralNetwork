from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# read the csv and replaces some string to integers
df = pd.read_csv("train.csv").replace({np.nan: -10, ' ': ''})
# all to lower
df = df.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)

df.columns = map(str.lower, df.columns)

# get all input collumn with string
columns_to_encode = df.drop(['advies', 'studentnummer', 'plaats', 'reden_stoppen', 'voorkeursopleiding'], axis=1).select_dtypes(
    include=['object']).columns.values

test = df.drop(['advies', 'studentnummer', 'plaats', 'reden_stoppen', 'voorkeursopleiding'], axis=1).select_dtypes(
    include=['int']).columns.values

print(columns_to_encode)

CATEGORICAL_COLUMNS = columns_to_encode
CONTINUOUS_COLUMNS = ["Age", "SibSp", "Parch", "Fare", "PassengerId", "Pclass"]

SURVIVED_COLUMN = "Advies"