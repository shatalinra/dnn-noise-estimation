import os, sys, logging, argparse, random

import tensorflow as tf
import matplotlib.pyplot as plt

import data
import noise_estimator
from models import chuah_et_al, simple, efficient

# parse command line args
parser = argparse.ArgumentParser(description='Noise estimation using few convolutional neural network')
parser.add_argument('--log', help='Path to a log file.')
parser.add_argument('--validate', action='store_true', help='Validate models based on image separate from training or testing data.')
script_args = parser.parse_args()

# setup logging before anything else
log_format = '%(asctime)s: <%(levelname)s> %(message)s'
if script_args.log:
    try:
        error_stream = logging.StreamHandler()
        error_stream.setLevel(logging.INFO)
        log_file = logging.FileHandler(script_args.log)
        logging.basicConfig(format=log_format, level=logging.INFO, handlers=[error_stream, log_file])
    except OSError as err:
        print("Error while creating log {}: {}. Exiting...".format(err.filename, err.strerror))
        input("Press Enter to continue...")
        sys.exit(1)
else:
    logging.basicConfig(format=log_format, level=logging.INFO)

# now we can setup hooks for uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

# giving message that log is indeed initialized
print("Log initialized")

# common procedure for training, evaluating and validating model
def try_model(name, patch_size, patch_stride, batch_size, preprocessing, model_trainer, validate):
    logging.info("Trying " + name + " model")
    estimator = noise_estimator.NoiseEstimator(patch_size, patch_stride, preprocessing, model_trainer)
    try:
        estimator.load("trained_models/" + name)

        if validate:
            # generate validation data using image which should not be in train or test set
            clean_image = data.load_image('../coco/2017/train/000000001955.jpg')
            noise_level =  random.randint(0, 9)
            validation_image = data.generate_image(clean_image, noise_level)

            # show original and noised image in order to check that noise generation is fine
            fig=plt.figure(figsize=(8, 2))
            fig.add_subplot(1, 2, 1)
            plt.imshow(clean_image)
            fig.add_subplot(1, 2, 2)
            plt.imshow(validation_image)
            plt.show()

            # classify noise level on the image
            logging.info("Case witn noise level %d", noise_level)
            classes, confidences = estimator(validation_image)
            for i in range(0, 10):
                logging.info("\tEstimated confidence in noise level %d: %.3f", classes[i], confidences[i])

        else:
            # generate testing data using portion of MS COCO 2017 train images
            dataset = data.NoisyDataset("../coco/2017/train/", 154, 359, patch_size, patch_stride, batch_size)

            # now evaluate accuracy
            accuracy = estimator.evaluate(dataset)
            logging.info("Accuracy is %0.1f%%", 100 * accuracy)

    except IOError:
        # looks like we don't have trained model, so we have to train one from scratch
        dataset = data.NoisyDataset("../coco/2017/train/", 9, 151, patch_size, patch_stride, batch_size)
        estimator.train(dataset, "trained_models/" + name)


# we start with trying models based on non-overlapping 32x32 patches which capture very little frame information
try_model("chuah_et_al", 32, 32, 256, None, chuah_et_al.train_model, script_args.validate)
try_model("simple", 32, 32, 100, None, simple.train_model, script_args.validate)

# now we try pretrained models using overlapping 224x224 patches which should capture a lot of visual information
try_model("efficent", 224, 224, 32, efficient.preprocess, efficient.train_model, script_args.validate)