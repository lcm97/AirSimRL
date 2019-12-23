import re
import pandas as pd
import os
import numpy as np
import utils
from keras import backend as K
from keras.preprocessing import image


class DroneDataGenerator(image.ImageDataGenerator):
    """
    Generate minibatches of images and labels with real-time augmentation.
    The only function that changes w.r.t. parent class is the flow that
    generates data. This function needed in fact adaptation for different
    directory structure and labels. All the remaining functions remain
    unchanged.
    For an example usage, see the evaluate.py script
    """
    def flow_from_directory(self, directory, color_mode='grayscale', batch_size=32,
            shuffle=True, seed=None, follow_links=False,):
        return DroneDirectoryIterator(
                directory, self,
                color_mode=color_mode,
                batch_size=batch_size, shuffle=shuffle, seed=seed,
                follow_links=follow_links,)

    def random_transform_with_states(self, x,):
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


class DroneDirectoryIterator(image.Iterator):
    """
    Class for managing data loading.of images and labels
    We assume that the folder structure is:
    root_folder/
           folder_1/
                    images/
                    sync_steering.txt or labels.txt
           folder_2/
                    images/
                    sync_steering.txt or labels.txt
           .
           .
           folder_n/
                    images/
                    sync_steering.txt or labels.txt
    # Arguments
       directory: Path to the root directory to read data from.
       image_data_generator: Image Generator.
       target_size: tuple of integers, dimensions to resize input images to.
       crop_size: tuple of integers, dimensions to crop input images.
       color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
       batch_size: The desired batch size
       shuffle: Whether to shuffle data or not
       seed : numpy seed to shuffle data
       follow_links: Bool, whether to follow symbolic links or not
    # TODO: Add functionality to save images to have a look at the augmentation
    """
    def __init__(self, directory, image_data_generator,
            color_mode='grayscale',
            batch_size=32, shuffle=True, seed=None, follow_links=False, resolution=32):
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.follow_links = follow_links
        self.resolution = resolution
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        if self.color_mode == 'rgb':
            self.image_shape = (self.resolution,self.resolution) + (3,)
        else:
            self.image_shape = (self.resolution,self.resolution) + (1,)

        # First count how many experiments are out there
        self.samples = 0

        experiments = []
        for subdir in sorted(os.listdir(directory)):
            if os.path.isdir(os.path.join(directory, subdir)):
                experiments.append(subdir)

        self.num_experiments = len(experiments)

        self.formats = {'pgm', 'png', 'jpg' }

        # Idea = associate each filename with a corresponding steering or label
        self.filenames = []
        self.ground_truth = []
        self.image_size = []

        for subdir in experiments:
            print(subdir)
            subpath = os.path.join(directory, subdir)
            self._decode_experiment_dir(subpath)

        # Conversion of list into array
        # self.ground_truth = np.array(self.ground_truth, dtype = K.floatx())

        assert self.samples > 0, "Did not find any data"

        print('Found {} images belonging to {} experiments.'.format(
                self.samples, self.num_experiments))
        super(DroneDirectoryIterator, self).__init__(self.samples,
                batch_size, shuffle, seed)

    def _recursive_list(self, subpath):
        return sorted(os.walk(subpath, followlinks=self.follow_links),
                key=lambda tpl: tpl[0])

    def _decode_experiment_dir(self, dir_subpath):
        # Load label in the experiment dir
        labels_filename = os.path.join(dir_subpath, "airsim_rec.txt")

        try:
            # ground_truth = np.loadtxt(labels_filename, usecols=None,
            #                       delimiter='\\t', skiprows=1)
            ground_truth = pd.read_csv(labels_filename, sep='\t',usecols=['Steering','Collision','Complexity'])

        except OSError as e:
            print("labels not found in dir {}".format(dir_subpath))
            raise IOError
        ground_truth = ground_truth.values
        ground_truth = np.expand_dims(ground_truth,axis=2)

        # Now fetch all images in the image subdir
        image_dir_path = os.path.join(dir_subpath, "images")
        for root, _, files in self._recursive_list(image_dir_path):
            #sort by image num
            sorted_files = sorted(files,
                    key = lambda fname: int(re.search(r'\d+',fname).group()))
            for frame_number, fname in enumerate(sorted_files):
                is_valid = False
                for extension in self.formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    absolute_path = os.path.join(root, fname)
                    self.filenames.append(os.path.relpath(absolute_path,
                            self.directory))
                    self.ground_truth.append(ground_truth[frame_number])
                    self.samples += 1


    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array) :
        """
        Public function to fetch next batch.
        # Returns
            The next batch of images and labels.
        """
        current_batch_size = index_array.shape[0]
        # Image transformation is not under thread lock, so it can be done in
        # parallel
        # batch_x = np.zeros((current_batch_size,) + self.image_shape,
        #         dtype=K.floatx())
        batch_x = []
        batch_y = np.zeros((current_batch_size, 3, 1),
                dtype=K.floatx())

        grayscale = self.color_mode == 'grayscale'

        # Build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            x = utils.load_img(os.path.join(self.directory, fname),
                    grayscale=grayscale)

            #x = self.image_data_generator.random_transform(x)
            transformed = self.image_data_generator.random_transform_with_states(x)
            x = transformed[0]
            x = self.image_data_generator.standardize(x)
            is_horize_flipped = transformed[1]
            # batch_x[i] = x
            batch_x.append(np.array(x))
            # Build batch of steering and collision data
            # batch_x = np.asarray(batch_x)
            if is_horize_flipped:
                self.ground_truth[index_array[i]][1] = self.ground_truth[index_array[i]][1]*-1
                batch_y[i] = self.ground_truth[index_array[i]]
            else:
                batch_y[i] = self.ground_truth[index_array[i]]

        #batch_y = [batch_steer, batch_coll]
        return batch_x, batch_y

# data_generator = DroneDataGenerator(rescale=1. / 255., horizontal_flip=True, brightness_range=[0.8, 1.2])
# train_generator = data_generator.flow_from_directory(directory='C:\dataset\AirSimData\data_raw',batch_size=32)
# sample_batch_train_data, sample_batch_label = next(train_generator)
# print(sample_batch_train_data)
# print(sample_batch_train_data[0].shape)
# print(np.array(sample_batch_train_data).shape)
