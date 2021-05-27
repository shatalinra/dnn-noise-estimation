from pathlib import Path

import tensorflow as tf
import logging

class NoiseEstimator(object):
    """wrapper for common functionality across noise estimation DNNs"""
    def __init__(self, patch_size, patch_stride, preprocessing, model_trainer, *args, **kwargs):
        self._patch_size = patch_size
        self._patch_stride = patch_stride
        self._preprocessing = preprocessing
        self._model = None
        self._model_trainer = model_trainer
        return super().__init__(*args, **kwargs)

    def load(self, path):
         self._model = tf.keras.models.load_model(path)

         # recompile model with metrics needed for evaluation
         self._model.compile(metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

         self._model.summary()

    def train(self, source_dataset, path):
        dir = Path(path)
        dir.mkdir(0o777, True, True)

        # for now preprocessing and tf.data.Dataset is needed only for model based on effecient net
        # and for  other models fitting all training data to GPU greatly helps
        patches = None
        labels = None
        dataset = None
        if self._preprocessing is None:
            patches, labels = source_dataset.data()
        else:
            dataset = source_dataset.dataset().map(self._preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.cache()

        init_attempts = 3
        best_model = None
        best_loss = tf.constant(100.0, dtype=tf.float32)
        for init_attempt in range(init_attempts):
            logging.info("Training model: attempt %d", init_attempt)
            if dataset is None:
                model, losses = self._model_trainer(patches, labels)
            else:
                model, losses = self._model_trainer(dataset)

            # save to log all metrics
            for epoch, loss in enumerate(losses):
                logging.info("Epoch %d, loss %.4f", epoch, losses[epoch])

            last_loss = losses[-1]
            if last_loss < best_loss:
                best_loss = last_loss
                best_model = model

        best_model.save(path)
        logging.info("Best loss is %.6f", best_loss)

        self._model = best_model

    def evaluate(self, source_dataset):
        # for now preprocessing and tf.data.Dataset is needed only for model based on effecient net
        # and for  other models fitting all training data to GPU greatly helps
        patches = None
        labels = None
        dataset = None
        if self._preprocessing is None:
            patches, labels = source_dataset.data()
        else:
            dataset = source_dataset.dataset().map(self._preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.cache()

        if dataset is None:
            # such small batch size slows down evaluation but it makes results more stable and accurate
            metrics = self._model.evaluate(patches, labels, batch_size=1, verbose = 0, return_dict=True)
        else:
            metrics = self._model.evaluate(dataset, verbose = 0, return_dict=True)

        return metrics["sparse_categorical_accuracy"]

    def __call__(self, image):
        # in order to estimate noise we break image on specified patches
        images = tf.expand_dims(image, 0)
        patches = tf.image.extract_patches(images, [1, self._patch_size, self._patch_size, 1], [1, self._patch_stride, self._patch_stride, 1], [1, 1, 1, 1], 'VALID')
        patches = tf.reshape(patches, [-1, self._patch_size, self._patch_size, 3])

        # preprocessing data if procedure is not none
        if self._preprocessing is not None:
            patches, labels = self._preprocessing(patches, None)

        # now we feed these patches to the model
        output = self._model(patches)

        # for each patch we are getting propabilities of each class but we need one estimate for whole image
        # lets try just summing those and then normalizing them once again
        sums = tf.reduce_sum(output, 0)
        sums = tf.reshape(sums, -1)
        total_sum = tf.reduce_sum(sums, 0)
        sums = sums / total_sum

        # now transform result into labels and their confidences
        classes = range(0, 10)
        sums = sums.numpy().tolist()
        sums, classes = zip(*sorted(zip(sums, classes), reverse=True))
        return classes, sums