from pathlib import Path
import tensorflow as tf
import random

# generate labeled training data based on source images from a specified folder
def generate_training_dataset(folder_path):
    patches = None
    labels = None

    for source_image_path in Path(folder_path).iterdir():

        # first we are loading the source image
        source_image_file = tf.io.read_file(str(source_image_path))
        source_image = tf.io.decode_jpeg(source_image_file)
        source_image = tf.expand_dims(source_image, 0)
        source_image = tf.cast(source_image, dtype = tf.float32) / 255

        # now we try different amount of noise
        for noise_level in range(0, 9):
            noise_ratio = noise_level / 10

            # each time generate new noise just to be sure
            noise = tf.random.normal(source_image.get_shape(), mean=0.5, stddev=1.0/6.0, dtype=tf.dtypes.float32)

            # there is still 5% probality that values may be outside of [0,1], so we are just clamping them
            noise = tf.clip_by_value(noise, 0, 1)

            # now we are blending source image with noise
            generated_image = (1.0 - noise_ratio) * source_image + noise_ratio * noise

            # now we are splitting image on patches
            new_patches = tf.image.extract_patches(generated_image, [1, 32, 32, 1], [1, 16, 16, 1], [1, 1, 1, 1], 'VALID')
            new_patches = tf.reshape(new_patches, [-1, 32, 32, 3])
            new_patches_count = new_patches.get_shape().as_list()[0]
            new_labels = tf.constant(noise_level, dtype=tf.int32, shape=(new_patches_count))

            # adding new data to output
            if patches is None:
                patches = new_patches
            else:
                patches = tf.concat([patches, new_patches], 0)
            if labels is None:
                labels = new_labels
            else:
                labels = tf.concat([labels, new_labels], 0)

    return patches, labels

def generate_testing_dataset(folder_path):
    patches = None
    labels = None

    for source_image_path in Path(folder_path).iterdir():

        # first we are loading the source image
        source_image_file = tf.io.read_file(str(source_image_path))
        source_image = tf.io.decode_jpeg(source_image_file)
        source_image = tf.expand_dims(source_image, 0)
        source_image = tf.cast(source_image, dtype = tf.float32) / 255

        # selecting noise level randomly
        noise_level = random.randint(0, 9)
        noise_ratio = noise_level / 10

        # each time generate new noise just to be sure
        noise = tf.random.normal(source_image.get_shape(), mean=0.5, stddev=1.0/6.0, dtype=tf.dtypes.float32)

        # there is still 5% probality that values may be outside of [0,1], so we are just clamping them
        noise = tf.clip_by_value(noise, 0, 1)

        # now we are blending source image with noise
        generated_image = (1.0 - noise_ratio) * source_image + noise_ratio * noise

        # now we are splitting image on patches
        new_patches = tf.image.extract_patches(generated_image, [1, 32, 32, 1], [1, 16, 16, 1], [1, 1, 1, 1], 'VALID')
        new_patches = tf.reshape(new_patches, [-1, 32, 32, 3])
        new_patches_count = new_patches.get_shape().as_list()[0]
        new_labels = tf.constant(noise_level, dtype=tf.int32, shape=(new_patches_count))

        # adding new data to output
        if patches is None:
            patches = new_patches
        else:
            patches = tf.concat([patches, new_patches], 0)
        if labels is None:
            labels = new_labels
        else:
            labels = tf.concat([labels, new_labels], 0)

    return patches, labels