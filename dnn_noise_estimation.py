import os, sys, logging, argparse
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import matplotlib.pyplot as plt
import data
import noise_estimator
import chuah_et_al

# разбираем входящие параметры
parser = argparse.ArgumentParser(description='Noise estimation DNN training script')
parser.add_argument('--log', help='Path to a log file.')
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

#tf.debugging.set_log_device_placement(True)


# generating training data
training_patches, training_labels = data.generate_training_dataset('train/')
logging.info("Training data size %d", training_labels.get_shape().as_list()[0])

# load or train Chuah et al model
chuah_et_all_estimator = noise_estimator.NoiseEstimator(chuah_et_al.get_model)
try:
    chuah_et_all_estimator.load("model")
except IOError:
    chuah_et_all_estimator.train(training_patches, training_labels, "model")

# generate test data
testing_patches, testing_labels = data.generate_training_dataset('test/')
logging.info("Actual noise level is %d", tf.reduce_mean(testing_labels))

# run it though estimator
output = chuah_et_all_estimator(testing_patches)
prob_sum = tf.reduce_sum(output, 0)
prob_sum = tf.reshape(prob_sum, -1)
logging.info("Estimated noise level is %d", tf.math.argmax(prob_sum))