import tensorflow as tf
import logging


backbone = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
backbone.trainable = False

def preprocess(patches, labels):
    return backbone(255 * patches, training = False), labels

def train_model(data, labels):
    model = tf.keras.Sequential(name = "efficient")
    model.add(tf.keras.Input(shape = [7, 7, 1280]))  # our data should be efficient embedding
    model.add(tf.keras.layers.GlobalAveragePooling2D(name="pool"))
    model.add(tf.keras.layers.Dropout(0.5, name="dropout"))
    model.add(tf.keras.layers.Dense(10, name="dense"))
    model.add(tf.keras.layers.Softmax(name = "softmax"))
    model.summary(print_fn=lambda x: logging.info(x))

    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(), optimizer=tf.optimizers.Adam(learning_rate = 0.001))

    history = model.fit(data, labels, 64, 400, verbose=2)
    return model, history.history["loss"]
