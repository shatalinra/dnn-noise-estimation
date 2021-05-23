# model taken from 2017 paper "Detection of Gaussian Noise and Its Level using Deep Convolutional Neural Network" by Chuah et al
# but paper does not contain which optimizer or loss. Moreover looking at training dataset size and number of model parameters
# it is very likely that researchers basically overfitted their model. In any case without details I cannot properly replicate their results,
# so it would be something more of the inspired by their paper than actual thing they did.

import tensorflow as tf
import logging

def train_model(patches, labels):
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
    model.summary(print_fn=lambda x: logging.info(x))

    # the paper states learning rate equal to 0.01 was used but that really depends on type of optimizer
    # for Adam it seems lesser values achieve actual results
    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(), optimizer=tf.optimizers.Adam(learning_rate = 0.0001))

    # the paper states that batch size of 100 and 100 epochs were used but I increased it to make learning more stable even though it slows down
    history = model.fit(patches, labels, 256, 500, verbose=2)
    return model, history.history["loss"]