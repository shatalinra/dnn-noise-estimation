# model taken from 2017 paper "Detection of Gaussian Noise and Its Level using Deep Convolutional Neural Network" by Chuah et al
# but paper does not contain which optimizer or loss. Moreover looking at training dataset size and number of model parameters
# it is very likely that researchers basically overfitted their model. In any case without details I cannot properly replicate their results,
# so it would be something more of the inspired by their paper than actual thing they did.

import tensorflow as tf
import logging, callbacks

def build_model():
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

def evaluate_model(model, patches, labels):
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    with tf.GradientTape() as tape:
        output = model(patches)
        loss = loss_function(labels, output)
        grad = tape.gradient(loss, model.trainable_variables)

    return loss, tf.linalg.global_norm(grad)

def train_model(patches, labels):
    model = build_model()
    model.summary(print_fn=lambda x: logging.info(x))

    # in paper they said about using 0.01 learning rate but that really depends on type of optimizer
    # for Adam it seems lesser values achieve actual results
    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(), optimizer=tf.optimizers.Adam(learning_rate = 0.0002))

    reporting = callbacks.ProgressLogging(evaluate_model, patches, labels, 5)
    history = model.fit(patches, labels, 100, 100, verbose=0, callbacks=[reporting])
    return model, history.history["loss"][-1]