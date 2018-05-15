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

# get the test set


def read_testset():
    # read the csv and replaces some string to integers
    df = pd.read_csv("test.csv").replace({np.nan: -10, ' ': ''})
    # all to lower
    df = df.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)

    df.columns = map(str.lower, df.columns)

    # get all input collumn with string
    columns_to_encode = df.drop(['advies', 'studentnummer', 'plaats', 'reden_stoppen', 'voorkeursopleiding'], axis=1).select_dtypes(
        include=['object']).columns.values
    # Converting categorical data into numbers
    df = pd.get_dummies(df, columns=columns_to_encode)

    df.columns = [c.replace(' ', '') for c in df.columns]
    df.columns = [c.replace('+', '') for c in df.columns]
    df.columns = [c.replace('&', '') for c in df.columns]
    df.columns = [c.replace(',', '') for c in df.columns]
    df.columns = [c.replace('(', '') for c in df.columns]
    df.columns = [c.replace(')', '') for c in df.columns]
    df.columns = [c.replace('%', '') for c in df.columns]
    df.columns = [c.replace("'", '') for c in df.columns]

    # all the values of the csv
    X = df[df.drop(['advies', 'studentnummer', 'plaats',
                    'reden_stoppen', 'voorkeursopleiding'], axis=1).columns[0:192]]

    # get the expected output
    y = df[df.columns[3]]
    # encode the output
    Y = pd.get_dummies(y)

    return(X, Y)

# get the data set


def read_dataset():
    # read the csv and replaces some string to integers
    df = pd.read_csv("train.csv").replace({np.nan: -10, ' ': ''})
    # all to lower
    df = df.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)

    df.columns = map(str.lower, df.columns)

    # get all input collumn with string
    columns_to_encode = df.drop(['advies', 'studentnummer', 'plaats', 'reden_stoppen', 'voorkeursopleiding'], axis=1).select_dtypes(
        include=['object']).columns.values
    # Converting categorical data into numbers
    df = pd.get_dummies(df, columns=columns_to_encode)

    df.rename(columns={
              'nadereori�ntatieopeenad-opleidingin�s-hertogenbosch': 'hertogenbosch'}, inplace=True)

    df.columns = [c.replace(' ', '') for c in df.columns]
    df.columns = [c.replace('+', '') for c in df.columns]
    df.columns = [c.replace('&', '') for c in df.columns]
    df.columns = [c.replace(',', '') for c in df.columns]
    df.columns = [c.replace('(', '') for c in df.columns]
    df.columns = [c.replace(')', '') for c in df.columns]
    df.columns = [c.replace('%', '') for c in df.columns]
    df.columns = [c.replace("'", '') for c in df.columns]

    # all the values of the csv
    X = df[df.drop(['advies', 'studentnummer', 'plaats',
                    'reden_stoppen', 'voorkeursopleiding'], axis=1).columns[0:192]]

    # get the expected output
    y = df[df.columns[3]]
    # encode the output
    Y = pd.get_dummies(y)

    return(X, Y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


# read the data set
train_x, train_y = read_dataset()

test_x, test_y = read_testset()

# Feature columns describe how to use the input.
my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# Build a DNN with 2 hidden layers and 10 nodes in each hidden layer.
classifier = tf.estimator.LinearClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 10 nodes each.
    optimizer=tf.train.AdamOptimizer(),
    hidden_units=[100, 100],
    # The model must choose between 3 classes.
    n_classes=3)

# Train the Model.
classifier.train(
    input_fn=lambda: train_input_fn(train_x, train_y, 32),
    steps=1000)

# # Evaluate the model.
# eval_result = classifier.evaluate(
#     input_fn=lambda:eval_input_fn(test_x, test_y, 32))

# print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
