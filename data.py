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

class NoisyDataset:
    """Dataset of noisy MS COCO training images with labels corresponding to levels of noise applied."""

    def __init__(self, path_prefix, start_id, end_id, patch_size, patch_stride, batch_size):
        self._path_prefix = path_prefix
        self._start_id = start_id
        self._end_id = end_id
        self._patch_size = patch_size
        self._patch_stride = patch_stride
        self._batch_size = batch_size

    def data_generator(self):
        # iteratve over ids and noise levels to generate patches with labels
        for id in range(self._start_id, self._end_id):
            image_path = Path(self._path_prefix + str(id).zfill(12) + ".jpg")

            # not all ids are included into MS COCO training set, so simply skip them
            if not image_path.exists():
                continue

            # MS COCO contains grayscale images, skip them
            source_image = load_image(image_path)
            if source_image.get_shape().as_list()[2] != 3:
                continue

            for noise_level in range(0, 10):
                generated_image = generate_image(source_image, noise_level)

                images = tf.expand_dims(generated_image, 0)

                sizes = [1, self._patch_size, self._patch_size, 1]
                strides = [1, self._patch_stride, self._patch_stride, 1]
                patches = tf.image.extract_patches(images,  sizes, strides, [1, 1, 1, 1], 'VALID')
                patches = tf.reshape(patches, [-1, self._patch_size, self._patch_size, 3])
                patches_count = patches.get_shape().as_list()[0]
                labels = tf.constant(noise_level, dtype=tf.int32, shape=(patches_count))
                yield patches, labels

    def data(self):
        """ Collect all data in tensors which allows fitting small dataset on GPU completely"""
        all_patches = None
        all_labels = None

        for patches, labels in self.data_generator():
            if all_patches is None:
                all_patches = patches
            else:
                all_patches = tf.concat([all_patches, patches], 0)
            if all_labels is None:
                all_labels = labels
            else:
                all_labels = tf.concat([all_labels, labels], 0)

        return all_patches, all_labels

    def dataset(self):
        """ Generate tf.data.Dataset which uses CPU for all its internal processing and allows creating large datasets"""

        def generator():
            # we need some caches in order to produce fixed size batches from variable amount of patches per image
            patch_cache = None
            label_cache = None

            # iterate over data and it to caches
            for patches, labels in self.data_generator():
                if patch_cache is None:
                    patch_cache = patches
                else:
                    patch_cache = tf.concat([patch_cache, patches], 0)
                if label_cache is None:
                    label_cache = labels
                else:
                    label_cache = tf.concat([label_cache, labels], 0)

                # now slice the cache on fixed-size batches
                total_count = label_cache.get_shape().as_list()[0]
                total_batches = total_count // self._batch_size
                for i in range(0, total_batches):
                    patch_slice = patch_cache[i*self._batch_size:(i+1)*self._batch_size]
                    label_slice = label_cache[i*self._batch_size:(i+1)*self._batch_size]
                    yield patch_slice, label_slice

                # now we should leave only last incomplete slice in caches
                patch_cache = patch_cache[total_batches*self._batch_size:total_count]
                label_cache = label_cache[total_batches*self._batch_size:total_count]

        patch_spec = tf.TensorSpec(shape=(self._batch_size, self._patch_size, self._patch_size, 3), dtype=tf.float32)
        label_spec = tf.TensorSpec(shape=(self._batch_size), dtype=tf.int32)
        return tf.data.Dataset.from_generator(generator, output_signature = (patch_spec, label_spec))