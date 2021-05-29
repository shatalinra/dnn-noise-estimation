# model taken from 2017 paper "Detection of Gaussian Noise and Its Level using Deep Convolutional Neural Network" by Chuah et al
# but the paper does not specify optimizer or loss used. Moreover looking at training dataset size and number of model parameters
# it is likely that researchers overfitted their model. In any case, without mentioned details it is impossible to properly replicate the results,
# so it would be something more of the inspired by their paper.

import tensorflow as tf
import logging

def train_model(patches, labels):
    model =  tf.keras.Sequential(name = "chuan_et_al")
    model.add(tf.keras.layers.Conv2D(input_shape=(32, 32, 3), filters=20, kernel_size = 5, strides = 1, name = "conv_1"))
    model.add(tf.keras.layers.MaxPool2D(pool_size = 2, name = "max_pool_1"))
    model.add(tf.keras.layers.Conv2D(filters=50, kernel_size = 5, strides = 1, name = "conv_2"))
    model.add(tf.keras.layers.MaxPool2D(pool_size = 2, name = "max_pool_2"))
    model.add(tf.keras.layers.Conv2D(filters=500, kernel_size = 4, strides = 1, name = "conv_3"))
    model.add(tf.keras.layers.ReLU(name = "conv_3_relu"))
    model.add(tf.keras.layers.Conv2D(filters=10, kernel_size = 2, strides = 1, name = "conv_4"))
    model.add(tf.keras.layers.Softmax(name = "conv_4_softmax"))
    model.summary(print_fn=lambda x: logging.info(x))

    # the paper states learning rate equal to 0.01 was used but that really depends on type of optimizer
    # for Adam it seems lesser values achieve best results
    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(), optimizer=tf.optimizers.Adam(learning_rate = 0.0001))

    # the paper states that batch size of 100 and 100 epochs were used but I increased it to make learning more stable even though it slows down
    history = model.fit(patches, labels, batch_size=256, epochs=500, verbose=2)
    return model, history.history["loss"]