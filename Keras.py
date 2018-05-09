from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

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
    X = df[df.drop(['advies', 'studentnummer', 'plaats', 'reden_stoppen', 'voorkeursopleiding'], axis=1).columns[0:192]]
    
    # get the expected output
    y = df[df.columns[3]]
    # encode the output
    # encoder = LabelEncoder()
    # encoder.fit(y)
    # y = encoder.transform(y).reshape(-1, 1)
    # Y = OneHotEncoder(sparse=False).fit_transform(y)
    Y = pd.get_dummies(y)
    
    return(X, Y)


# read the data set
X, Y = read_dataset()

# split the dataset into train and test part
train_x, test_x, train_y, test_y = train_test_split(
    X, Y, test_size=0.13986013986013987)

# create model
model = Sequential()
model.add(Dense(12, input_dim=80, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

# the optimizer
adam = Adam(lr=0.01)
sgd = SGD(lr=0.01)
rmsprop = RMSprop(lr=0.01)

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['binary_accuracy', 'categorical_accuracy'])

# Fit/train the model
model.fit(train_x, train_y, epochs=500, shuffle=False, batch_size=10, validation_data=(test_x, test_y))

# evaluate the model
scores = model.evaluate(test_x, test_y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("\n%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))

# # calculate predictions
# predictions = model.predict(test_x)
# # round predictions
# rounded = [round(x[0]) for x in predictions]
# print(rounded)

'''
010 positief
100 negatief
001 twijfel
'''
