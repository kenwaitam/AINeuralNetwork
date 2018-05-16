from __future__ import absolute_import, division, print_function

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# read the csv and replaces some string to integers
df = pd.read_csv("intake2.csv").replace({' ': ''})

df = df.fillna(-50)

# all to lower
df = df.apply(lambda x: x.str.lower() if (x.dtype == 'object') else x)

df.columns = map(str.lower, df.columns)

df.columns = [c.replace(' ', '') for c in df.columns]
df.columns = [c.replace('+', '') for c in df.columns]
df.columns = [c.replace('&', '') for c in df.columns]
df.columns = [c.replace(',', '') for c in df.columns]
df.columns = [c.replace('(', '') for c in df.columns]
df.columns = [c.replace(')', '') for c in df.columns]
df.columns = [c.replace('%', '') for c in df.columns]
df.columns = [c.replace("'", '') for c in df.columns]

df['gewogen_gemiddelde'] = df['gewogen_gemiddelde'].astype(int)

# get all input collumn with string
categorical = df.drop(['studentnummer', 'plaats', 'reden_stoppen', 'voorkeursopleiding'],
                      axis=1).select_dtypes(
    include=['object']).columns.values

continuous = df.drop(['studentnummer', 'jaar', 'plaats', 'reden_stoppen', 'voorkeursopleiding'],
                     axis=1).select_dtypes(
    include=['integer']).columns.values

CATEGORICAL_COLUMNS = categorical
CONTINUOUS_COLUMNS = continuous

for col in CATEGORICAL_COLUMNS:
    df[col] = df[col].astype('category')

cat_columns = df.select_dtypes(['category']).columns

df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

cols_with_int8 = df.select_dtypes(['int8']).columns

for col in cols_with_int8:
    df[col] = df[col].astype('int64')

y = df.pop('advies')

# all the values of the csv
X = df[df.drop(['studentnummer', 'plaats', 'jaar',
                'reden_stoppen', 'voorkeursopleiding'], axis=1).columns[0:192]]

# split the dataset into train and test part
train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.13986013986013987, shuffle=False)

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.repeat().batch(batch_size)

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


# Feature columns describe how to use the input.
my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# Build a DNN with 2 hidden layers and 10 nodes in each hidden layer.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 10 nodes each.
    optimizer=tf.train.AdamOptimizer(),
    hidden_units=[10, 10, 10],
    # The model must choose between 3 classes.
    n_classes=3)

tf.logging.set_verbosity(tf.logging.INFO)

# Train the Model.
classifier.train(
    input_fn=lambda: train_input_fn(train_x, train_y, 32),
    steps=10000)

# Evaluate the model.
eval_result = classifier.evaluate(
    input_fn=lambda: eval_input_fn(test_x, test_y, 32))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

expected = ['Positief']

SPECIES = ['Negatief', 'Positief', 'Twijfel']

predict_x = {
    'naam_vooropleiding': [3],
    'geslacht': [0],
    'was_aanwezig': [0],
    'aantal_weigeringen': [0],
    'gewogen_gemiddelde': [16],
    'komt_studeren': [1],
    'competenties': [5],
    'capaciteiten': [7],
    'intr.motivatie': [7],
    'extr.motivatie': [6],
    'is_mbo_deficient': [1],
    'ondernemenincombinatiemetstudie': [0],
    'nadereorientatieopeenad-opleidingins-hertogenbosch': [0],
    'nadereorientatieopeenadopleidinginroosendaal': [0],
    'persoonlijkbijspijker-advies': [0],
    'functiebeperkingnamelijk': [1],
    'dyslexie': [0],
    'topsportincombinatiemetstudie': [0],
    'chronischeziekte': [0],
    'problemenindepersoonlijkesfeer': [0],
    'anderonderwerpnamelijk': [1],
    'financieleproblemeninrelatietotstuderen': [0],
    'studiekeuze': [0],
    'extragesprekmetopleiding': [0],
    'bijscholingengels': [0],
    'peermentoringspecifiekterondersteuningvanassadhd': [1],
    'aanmeldenvoorverkortopleidingstraject': [0],
    'hyperrr-peermentoringspecifiekdooropleidingaangeboden': [0],
    'mbo-oefenennederlandsetaal': [0],
    'mbo-peermentoringspecifiekterondersteuningvanassadhd': [0],
    'mbo-peermentoringterondersteuninginhetalgemeendoorhogerejaarsstudentvandezelfdeopleiding': [0],
    'trajectopleiding': [2],
}

predictions = classifier.predict(
    input_fn=lambda: eval_input_fn(predict_x, labels=None,
                                   batch_size=32))

template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print(template.format(SPECIES[class_id],
                          100 * probability, expec))
