import tensorflow as tf
import logging

def train_model(patches, labels):
    model =  tf.keras.Sequential(name = "simple")
    model.add(tf.keras.layers.Conv2D(input_shape=(32, 32, 3), filters=8, kernel_size = 2, strides = 2, name = "conv_1"))
    model.add(tf.keras.layers.LeakyReLU(alpha = 0.1, name = "conv_1_leaky"))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size = 2, strides = 2, name = "conv_2"))
    model.add(tf.keras.layers.LeakyReLU(alpha = 0.1, name = "conv_2_leaky"))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size = 2, strides = 2, name = "conv_3"))
    model.add(tf.keras.layers.LeakyReLU(alpha = 0.1, name = "conv_3_leaky"))
    model.add(tf.keras.layers.Conv2D(filters=1, kernel_size = 2, strides = 2, name = "conv_4"))
    model.add(tf.keras.layers.LeakyReLU(alpha = 0.1, name = "conv_4_leaky"))
    model.add(tf.keras.layers.Conv2D(filters=10, kernel_size = 2, strides = 2, name = "conv_5"))
    model.add(tf.keras.layers.LeakyReLU(alpha = 0.1, name = "conv_5_leaky"))
    model.add(tf.keras.layers.Softmax(name = "conv_5_softmax"))
    model.summary(print_fn=lambda x: logging.info(x))

    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(), optimizer=tf.optimizers.Adam(learning_rate = 0.004))

    history = model.fit(patches, labels, batch_size=100, epochs=300, verbose=2)
    return model, history.history["loss"]
