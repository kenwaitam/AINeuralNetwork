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
df = df.apply(lambda x: x.str.lower() if (x.dtype == 'object') else x)

df.columns = map(str.lower, df.columns)

df['gewogen_gemiddelde'] = df['gewogen_gemiddelde'].astype(int)

# get all input collumn with string
categorical = df.drop(['advies', 'studentnummer', 'plaats', 'reden_stoppen', 'voorkeursopleiding'],
                      axis=1).select_dtypes(
    include=['object']).columns.values

continuous = df.drop(['advies', 'studentnummer', 'jaar', 'plaats', 'reden_stoppen', 'voorkeursopleiding'],
                     axis=1).select_dtypes(
    include=['integer']).columns.values

CATEGORICAL_COLUMNS = categorical
CONTINUOUS_COLUMNS = continuous

print(CATEGORICAL_COLUMNS)

SURVIVED_COLUMN = "advies"

sex = tf.feature_column.categorical_column_with_hash_bucket(
    "geslacht", hash_bucket_size=1000)

aanw = tf.feature_column.categorical_column_with_hash_bucket(
    "was_aanwezig", hash_bucket_size=1000)

come = tf.feature_column.categorical_column_with_hash_bucket(
    "komt_studeren", hash_bucket_size=1000)

mbo_deficient = tf.feature_column.categorical_column_with_hash_bucket(
    "is_mbo_deficient", hash_bucket_size=1000)

# traject = tf.feature_column.categorical_column_with_hash_bucket(column_name="traject opleiding",
#                                                     keys=["verkort traject", "regulier traject"])

voor = tf.feature_column.categorical_column_with_hash_bucket(
    "naam_vooropleiding", hash_bucket_size=1000)

# aantal_weigeringen = tf.contrib.layers.real_valued_column("aantal_weigeringen")
# gewogen_gemiddelde = tf.contrib.layers.real_valued_column("gewogen_gemiddelde")
# competenties = tf.contrib.layers.real_valued_column("competenties")
# capaciteiten = tf.contrib.layers.real_valued_column("capaciteiten")
# intr = tf.contrib.layers.real_valued_column("intr. motivatie")
# extr = tf.contrib.layers.real_valued_column("extr. motivatie")
# ondernemen_in_combinatie_met_studie = tf.contrib.layers.real_valued_column("ondernemen in combinatie met studie")
# nadere_orientatie_op_een_ad_opleiding_in_shertogenbosch = tf.contrib.layers.real_valued_column(
#     "nadere orientatie op een ad-opleiding in s-hertogenbosch")
# nadere_orientatie_op_een_ad_opleiding_in_roosendaal = tf.contrib.layers.real_valued_column(
#     "nadere orientatie op een ad opleiding in roosendaal")
# persoonlijk_bijspijker_advies = tf.contrib.layers.real_valued_column("persoonlijk 'bijspijker'-advies")
# functiebeperking_namelijk = tf.contrib.layers.real_valued_column("functiebeperking, namelijk")
# dyslexie = tf.contrib.layers.real_valued_column("dyslexie")
# topsport_in_combinatie_met_studie = tf.contrib.layers.real_valued_column("topsport in combinatie met studie")
# chronische_ziekte = tf.contrib.layers.real_valued_column("chronische ziekte")
# problemen_in_de_persoonlijke_sfeer = tf.contrib.layers.real_valued_column("problemen in de persoonlijke sfeer")
# ander_onderwerp_namelijk = tf.contrib.layers.real_valued_column("ander onderwerp, namelijk")
# financiele_problemen_in_relatie_tot_studeren = tf.contrib.layers.real_valued_column(
#     "financiele problemen in relatie tot studeren")
# studiekeuze = tf.contrib.layers.real_valued_column("studiekeuze")
# extra_gesprek_met_opleiding = tf.contrib.layers.real_valued_column("extra gesprek met opleiding")
# bijscholing_engels = tf.contrib.layers.real_valued_column("bijscholing engels")
# peermentoring = tf.contrib.layers.real_valued_column("peermentoring (specifiek ter ondersteuning van ass & ad(h)d)")
# aanmelden_voor_verkort_opleidingstraject = tf.contrib.layers.real_valued_column(
#     "aanmelden voor verkort opleidingstraject")
# hyperrr_peermentoring = tf.contrib.layers.real_valued_column(
#     " hyperrr - peermentoring (specifiek door opleiding aangeboden)")
# oefenen_nederlandse = tf.contrib.layers.real_valued_column("mbo - oefenen nederlandse taal")
# peermentoring_adhd = tf.contrib.layers.real_valued_column(
#     "mbo - peermentoring (specifiek ter ondersteuning van ass & ad(h)d)")
# peermentoring_opleiding = tf.contrib.layers.real_valued_column(
#     "mbo - peermentoring (ter ondersteuning in het algemeen door hogerejaars student van dezelfde opleiding)")

deep_columns = [
    sex,
    aanw,
    come,
    mbo_deficient,
    # tf.contrib.layers.embedding_column(traject, dimension=8),
    voor,
    # aantal_weigeringen,
    # gewogen_gemiddelde,
    # competenties,
    # capaciteiten,
    # intr,
    # extr,
    # ondernemen_in_combinatie_met_studie,
    # nadere_orientatie_op_een_ad_opleiding_in_shertogenbosch,
    # nadere_orientatie_op_een_ad_opleiding_in_roosendaal,
    # persoonlijk_bijspijker_advies,
    # functiebeperking_namelijk,
    # dyslexie,
    # topsport_in_combinatie_met_studie,
    # chronische_ziekte,
    # problemen_in_de_persoonlijke_sfeer,
    # ander_onderwerp_namelijk,
    # financiele_problemen_in_relatie_tot_studeren,
    # studiekeuze,
    # extra_gesprek_met_opleiding,
    # bijscholing_engels,
    # peermentoring,
    # aanmelden_voor_verkort_opleidingstraject,
    # hyperrr_peermentoring,
    # oefenen_nederlandse,
    # peermentoring_adhd,
    # peermentoring_opleiding,
]

# Feature columns describe how to use the input.
my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# Build a DNN with 2 hidden layers and 10 nodes in each hidden layer.
classifier = tf.estimator.DNNClassifier(
    feature_columns=deep_columns,
    # Two hidden layers of 10 nodes each.
    optimizer=tf.train.AdamOptimizer(),
    hidden_units=[100, 100],
    # The model must choose between 3 classes.
    n_classes=3)