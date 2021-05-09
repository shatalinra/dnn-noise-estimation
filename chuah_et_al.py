# model taken from 2017 paper "Detection of Gaussian Noise and Its Level using Deep Convolutional Neural Network" by Chuah et al

import tensorflow as tf

def get_model():
    model =  tf.keras.Sequential(name = "chuan_et_al")
    model.add(tf.keras.Input(shape = [32, 32, 3]))
    model.add(tf.keras.layers.Conv2D(filters=20, kernel_size = 5, strides = 1, name = "conv-1"))
    model.add(tf.keras.layers.MaxPool2D(pool_size = 2, name = "max-pool-1"))
    model.add(tf.keras.layers.Conv2D(filters=50, kernel_size = 5, strides = 1, name = "conv-2"))
    model.add(tf.keras.layers.MaxPool2D(pool_size = 2, name = "max-pool-2"))
    model.add(tf.keras.layers.Conv2D(filters=500, kernel_size = 4, strides = 1, name = "conv-3"))
    model.add(tf.keras.layers.ReLU(name = "conv-3-relu"))
    model.add(tf.keras.layers.Conv2D(filters=10, kernel_size = 2, strides = 1, name = "conv-4"))
    model.add(tf.keras.layers.Softmax(name = "conv-4-softmax"))
    return model
