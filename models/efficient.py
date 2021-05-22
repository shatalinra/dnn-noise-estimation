
import tensorflow as tf
import logging

def build_model():
    backbone = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(224, 224, 3))
    backbone.trainable = False

    model =  tf.keras.Sequential(name = "efficient")
    model.add(backbone)
    model.add(tf.keras.layers.GlobalAveragePooling2D(name="pool"))
    model.add(tf.keras.layers.Dense(10, name="dense"))
    model.add(tf.keras.layers.Softmax(name = "softmax"))
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

    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(), optimizer=tf.optimizers.RMSprop())

    min_loss_change = tf.constant(0.01, dtype=tf.float32)
    reporting = callbacks.ProgressLogging(evaluate_model, patches, labels, 5)

    history = model.fit(patches, labels, 1, 400, verbose=0, callbacks=[reporting])
    return model, history.history["loss"][-1]
