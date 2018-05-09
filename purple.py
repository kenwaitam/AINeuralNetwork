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

tf.enable_eager_execution()

# get the data set
def read_dataset():
    # read the csv and replaces some string to integers
    df = pd.read_csv("train.csv").replace(np.nan, -10)
    # all to lower
    df = df.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)

    # get all input collumn with string
    columns_to_encode = df.drop(['advies', 'studentnummer', 'plaats', 'reden_stoppen', 'voorkeursopleiding'], axis=1).select_dtypes(
        include=['object']).columns.values
    # Converting categorical data into numbers
    df = pd.get_dummies(df, columns=columns_to_encode)

    # all the values of the csv
    X = df[df.drop(['advies', 'studentnummer', 'plaats', 'reden_stoppen', 'voorkeursopleiding'], axis=1).columns[0:192]]
    
    # get the expected output
    y = df[df.columns[3]]
    # encode the output
    Y = pd.get_dummies(y)
    
    return(X, Y)

# read the data set
x, y = read_dataset()
