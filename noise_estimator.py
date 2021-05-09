from pathlib import Path

import tensorflow as tf
import logging

class ProgressLogging(tf.keras.callbacks.Callback):
    '''Custom callback for reporting training progress'''
    def __init__(self, input, labels, test_rate):
        '''
        display: Number of epochs to wait before outputting loss
        '''
        self.input = input
        self.labels = labels
        self.test_rate = test_rate
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.last_loss = None

    def on_epoch_end(self, epoch, logs = None):
        if epoch % self.test_rate != 0: return

        with tf.GradientTape() as tape:
            output = self.model(self.input)
            loss = self.loss(self.labels, output)
            grad = tape.gradient(loss, self.model.trainable_variables)

        delta = loss
        if self.last_loss:
            delta = self.last_loss - loss
        self.last_loss = loss

        logging.info('Epoch %d, loss %0.6f, change %0.6f, grad norm %0.6f, lr %0.6f', epoch, loss, delta, tf.linalg.global_norm(grad), self.model.optimizer.lr)

class NoiseEstimator(object):
    """wrapper for common functionality across noise estimation DNNs"""
    def __init__(self, model_generator, *args, **kwargs):
        self._model = None
        self._model_generator = model_generator
        return super().__init__(*args, **kwargs)

    def load(self, path):
         self._model = tf.keras.models.load_model(path)
         self._model.summary()

    def train(self, patches, labels, path):
        dir = Path(path)
        dir.mkdir(0o777, True, True)

        init_attempts = 3
        best_model = None
        best_loss = tf.constant(100.0, dtype=tf.float32)
        min_loss_change = tf.constant(0.000001, dtype=tf.float32)
        for init_attempt in range(init_attempts):
            model = self._model_generator()
            logging.info("Training model: attempt %d", init_attempt)
            loss = self._train_model(model, patches, labels, min_loss_change, 2500)
            if loss < best_loss:
                best_loss = loss
                best_model = model

        best_model.save(path)
        logging.info("Best loss is %.6f", best_loss)

        self._model = best_model

    def _train_model(self, model, patches, labels, min_loss_change, max_epoch):
        model.summary(print_fn=lambda x: logging.info(x))
        model.compile(loss=tf.losses.SparseCategoricalCrossentropy(), optimizer=tf.optimizers.Adam(learning_rate = 0.001))

        callback = ProgressLogging(patches, labels, 20)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.9, patience=10, min_lr=0.0001, min_delta=min_loss_change)
        stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100, mode='min', min_delta=min_loss_change)
        history = model.fit(patches, labels, 2048, max_epoch, verbose=0, callbacks=[callback, reduce_lr, stopping])
        return history.history["loss"][-1]

    def __call__(self, data):
        return self._model(data)