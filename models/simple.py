import tensorflow as tf
import logging

def build_model():
    model =  tf.keras.Sequential(name = "simple")
    model.add(tf.keras.Input(shape = [32, 32, 3]))
    model.add(tf.keras.layers.Conv2D(filters=8, kernel_size = 2, strides = 2, name = "conv-1"))
    model.add(tf.keras.layers.LeakyReLU(alpha = 0.1, name = "conv-1-leaky"))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size = 2, strides = 2, name = "conv-2"))
    model.add(tf.keras.layers.LeakyReLU(alpha = 0.1, name = "conv-2-leaky"))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size = 2, strides = 2, name = "conv-3"))
    model.add(tf.keras.layers.LeakyReLU(alpha = 0.1, name = "conv-3-leaky"))
    model.add(tf.keras.layers.Conv2D(filters=1, kernel_size = 2, strides = 2, name = "conv-4")) # last best was 1 with 93.2% but 4 got 91.7%
    model.add(tf.keras.layers.LeakyReLU(alpha = 0.1, name = "conv-4-leaky"))
    model.add(tf.keras.layers.Conv2D(filters=10, kernel_size = 2, strides = 2, name = "conv-5"))
    model.add(tf.keras.layers.LeakyReLU(alpha = 0.1, name = "conv-5-leaky"))
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

    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(), optimizer=tf.optimizers.Adam(learning_rate = 0.004))

    min_loss_change = tf.constant(0.01, dtype=tf.float32)
    reporting = callbacks.ProgressLogging(evaluate_model, patches, labels, 5)

    history = model.fit(patches, labels, 100, 300, verbose=0, callbacks=[reporting])
    return model, history.history["loss"][-1]
