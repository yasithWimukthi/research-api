import numpy
import tensorflow as tf
from flask import request
from tensorflow.python.keras.models import load_model


def predict():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(50, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(50, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(50, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(7, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model = tf.keras.models.load_model('model.h5')

    prediction = model.predict(numpy.array([[1,2,0,0,0,0,1,0,1,0,1,0,1,1,1,0,0,0,0,1,0,1,0,1,0,1,0,0,1,0,0,1,0,0,1,1,0,1,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,1,1,0,1,1,1,1]]))
    print(str(numpy.argmax(prediction)))


predict()

#[1,0,1,1,0,0,1,1,1,1,0,1,0,0,1,1,1,1,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,0,0,1,0,1,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,1,0,1,0,0,0,0]
#[1,2,0,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,1,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,1,0,1,0,0,0,0]