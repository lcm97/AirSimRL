from keras.preprocessing import image
import numpy as np
import keras.backend as K
import os

class DriveDataGenerator(image.ImageDataGenerator):

    def flow(self,
             x_images,
             y_steering=None,
             y_collision=None,
             y_complexity=None,
             batch_size=32,shuffle=True,seed=None,save_to_dir=None,save_prefix='',save_format='png',
             zero_drop_percentage=0.5
             ):
        return DriveIterator(
            x_images,y_steering,y_collision,y_complexity,self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            zero_drop_percentage=zero_drop_percentage,
        )
    def random_transform_with_states(self, x, seed=None):
        """Randomly augment a single image tensor.
        # Arguments
            x: 3D tensor, single image.
            seed: random seed.
        # Returns
            A tuple. 0 -> randomly transformed version of the input (same shape). 1 -> true if image was horizontally flipped, false otherwise
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis
        img_col_axis = self.col_axis
        img_channel_axis = self.channel_axis

        is_image_horizontally_flipped = False

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                     [0, np.cos(shear), 0],
                                     [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = image.transform_matrix_offset_center(transform_matrix, h, w)
            x = image.apply_transform(x, transform_matrix, img_channel_axis,
                                      fill_mode=self.fill_mode, cval=self.cval)

        if self.channel_shift_range != 0:
            x = image.random_channel_shift(x,
                                           self.channel_shift_range,
                                           img_channel_axis)
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = image.image.flip_axis(x, img_col_axis)
                is_image_horizontally_flipped = True

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = image.image.flip_axis(x, img_row_axis)

        return x, is_image_horizontally_flipped

class DriveIterator(image.Iterator):
    """Iterator yielding data from a Numpy array.
    # Arguments
        x: Numpy array of input data.
        y: Numpy array of targets data.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """
    def __init__(self, x_images, y_steering, y_collision, y_complexity, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png', zero_drop_percentage=0.5):
        if y_steering is not None and len(x_images) != len(y_steering):
            raise ValueError ('X and y (labels)'
                              'should have the same length'
                              'Found: X.shape = %s, y.shape = %s' %
                              (np.asarray(x_images).shape, np.asarray(y_steering).shape))
        if data_format is None:
            data_format = K.image_data_format()

        self.x_images = x_images
        self.zero_drop_percentage = zero_drop_percentage

        if self.x_images.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should ave rank 4. You passed an array '
                             'with shape', self.x_images.shape)
        channels_axis = 3 if data_format == 'channels_last' else 1
        if self.x_images.shape[channels_axis] not in {1, 3, 4}:
            raise ValueError('NumpyArrayIterator is set to use the '
                             'data format convention "' + data_format + '" '
                                                                        '(channels on axis ' + str(
                channels_axis) + '), i.e. expected '
                                 'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
                                                                                             'However, it was passed an array with shape ' + str(
                self.x_images.shape) +
                             ' (' + str(self.x_images.shape[channels_axis]) + ' channels).')

        if y_steering is not  None:
            self.y_steering  = y_steering
        else:
            self.y_steering = None

        if y_collision is not None:
            self.y_collision = y_collision
        else:
            self.y_collision = None

        if y_complexity is not  None:
            self.y_complexity  = y_complexity
        else:
            self.y_complexity = None

        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.batch_size = batch_size
        super(DriveIterator, self).__init__(x_images.shape[0], batch_size, shuffle, seed)

    def next(self):
        """
        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel

        return self.__get_indexes(index_array)

    def __get_indexes(self, index_array):
        index_array = sorted(index_array)
        if self.x_images is not None:
            batch_x_images = np.zeros(tuple([self.batch_size] + list(self.x_images.shape)[1:]), dtype=K.floatx())
        else:
            batch_x_images = None

        used_indexes = []
        is_horiz_flipped = []

        for i, j in enumerate(index_array):
            x_images = self.x_images[j]

            transformed = self.image_data_generator.random_transform_with_states(x_images.astype(K.floatx()))

            x_images = transformed[0]
            is_horiz_flipped.append(transformed[1]) #判断与之对应的image有没有被翻转
            x_images = self.image_data_generator.standardize(x_images)
            batch_x_images[i] = x_images

            used_indexes.append(j)
        batch_x= np.asarray(batch_x_images)

        if self.save_to_dir:
            for i in range(0, self.batch_size, 1):
                hash = np.random.randint(1e4)

                img = image.array_to_img(batch_x_images[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=1,
                                                                  hash=hash,
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        batch_y_steering = self.y_steering[list(sorted(used_indexes))]
        batch_y_collision = self.y_collision[list(sorted(used_indexes))]
        batch_y_complexity = self.y_complexity[list(sorted(used_indexes))]

        idx = []

        for i in range(0, len(is_horiz_flipped), 1):
            if batch_y_steering.shape[1] == 1:
                if is_horiz_flipped[i]:
                    batch_y_steering[i] *= -1

                if np.isclose(batch_y_steering[i], 0):
                    if np.random.uniform(low=0, high=1) < self.zero_drop_percentage:
                        idx.append(True)
                    else:
                        idx.append(False)
                else:
                    idx.append(True)
            else:
                if batch_y_steering[i][int(len(batch_y_steering[i]) / 2)] == 1:
                    if np.random.uniform(low=0, high=1) > self.zero_drop_percentage:
                        idx.append(True)
                    else:
                        idx.append(False)
                else:
                    idx.append(True)

                if is_horiz_flipped[i]:
                    batch_y_steering[i] = batch_y_steering[i][::-1]
        batch_y = [batch_y_steering, batch_y_collision, batch_y_complexity]

        batch_x = batch_x[idx]
        batch_y[0] = batch_y[0][idx]
        batch_y[1] = batch_y[1][idx]
        batch_y[2] = batch_y[2][idx]

        return batch_x, batch_y

    def _get_batches_of_transformed_samples(self, index_array):
        return self.__get_indexes(index_array)