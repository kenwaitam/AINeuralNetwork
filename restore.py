import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# convert excel to csv
# data_xls = pd.read_excel('test.xlsx', index_col=None)
# data_xls.to_csv('intake.csv', encoding='utf-8', index=False)

# get the data set
def read_dataset():
    # read the csv and replaces some string to integers
    df = pd.read_csv("intake.csv").replace(np.nan, -10)
    # all to lower
    df = df.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)

    # get all input collumn with string
    columns_to_encode = df.drop(['advies', 'studentnummer', 'plaats', 'reden_stoppen', 'voorkeursopleiding'], axis=1).select_dtypes(
        include=['object']).columns.values
    # Converting categorical data into numbers
    df = pd.get_dummies(df, columns=columns_to_encode)

    # all the values of the csv
    X = df[df.drop(['advies', 'studentnummer', 'plaats', 'reden_stoppen', 'voorkeursopleiding'], axis=1).columns[0:192]].values

    # get the expected output
    y1 = df[df.columns[3]]

    # encode the output
    # encoder = LabelEncoder()
    # encoder.fit(y)
    # y = encoder.transform(y).reshape(-1, 1)
    # Y = OneHotEncoder(sparse=False).fit_transform(y)
    Y = pd.get_dummies(y1).values

    return(X, Y, y1)


# read the data set
X, Y, y1 = read_dataset()

# define the learning rate
learning_rate = 0.1

# define the minimum iteration to minimize the error
training_epochs = 500

# the cost history
cost_history = np.empty(shape=[1], dtype=float)

# shape of X / number of column
n_dim = X.shape[1]
print("n_dim", n_dim)

# number of class / number of diffrent output
#Positief, Twijfel, Negatief
n_class = 3

# place to store model
model_path = "./NMI"


# define the number of hidden layers and number of neurons for each layer
n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

# the input value place holder
x = tf.placeholder(tf.float32, [None, n_dim])
print(x)

# the weight?
W = tf.Variable(tf.zeros([n_dim, n_class]))

# the biases?
b = tf.Variable(tf.zeros([n_class]))

# output of the model that we already know
y_ = tf.placeholder(tf.float32, [None, n_class])

# define the model
def multilayer_pereptron(x, weights, biases):

    # hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    # hidden layer with sigmoid activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    # hidden layer with sigmoid activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    # hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    # output with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer


# define the weights and the biases for each layer
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_3, n_class]))
}

biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_class]))
}

# initialize all the variables
init = tf.global_variables_initializer()

# save the model
saver = tf.train.Saver()

# Call model defined
y = multilayer_pereptron(x, weights, biases)

# define the cost function(lost function) and optimizer
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# create a session object and run it
sess = tf.Session()
sess.run(init)

saver.restore(sess, model_path)

prediction = tf.argmax(y, 1)

correct_prediction = tf.equal(prediction, tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(246, 285):
    prediction_run = sess.run(prediction, feed_dict={x: X[i].reshape(1, 80)})
    accuracy_run = sess.run(accuracy, feed_dict={x: X[i].reshape(1, 80), y_: Y})
    print('Original: ', y1[i], 'Prediction: ', prediction_run, 'Accuracy: ', accuracy_run)

sess.close()