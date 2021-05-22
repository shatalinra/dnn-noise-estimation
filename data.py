from pathlib import Path
import tensorflow as tf

def load_image(path):
    image_file = tf.io.read_file(str(path))
    image = tf.io.decode_jpeg(image_file)
    image = tf.cast(image, dtype = tf.float32) / 255
    return image

def generate_image(source_image, noise_level):
    noise_ratio = noise_level / 10

    # generating noise so that majority of values would be in [0,1] and rest are just clamped
    noise = tf.random.normal(source_image.get_shape(), mean=0.5, stddev=1.0/6.0, dtype=tf.dtypes.float32)
    noise = tf.clip_by_value(noise, 0, 1)

    # now we just blend noise in
    generated_image = (1.0 - noise_ratio) * source_image + noise_ratio * noise
    return generated_image

def generate_dataset(path_prefix, image_indices_path, patch_size, patch_stride):
    patches = None
    labels = None

    # first read image indices from a text file
    with open(image_indices_path) as indice_file:
        image_indices = [line.rstrip() for line in indice_file]

    # now generate image paths based on that
    image_paths = [path_prefix + x + ".jpg" for x in image_indices]

    # load source images and generate images with different amount of noise
    for source_image_path in image_paths:
        source_image = load_image(source_image_path)
        for noise_level in range(0, 10):
            generated_image = generate_image(source_image, noise_level)

            # now we are splitting image on patches
            images = tf.expand_dims(generated_image, 0)
            new_patches = tf.image.extract_patches(images, [1, patch_size, patch_size, 1], [1, patch_stride, patch_stride, 1], [1, 1, 1, 1], 'VALID')
            new_patches = tf.reshape(new_patches, [-1, patch_size, patch_size, 3])
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