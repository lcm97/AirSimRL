import random
from PIL import Image
import numpy as np
import pandas as pd
import sys
import os
import errno
import h5py


def checkAndCreateDir(full_path):
    """Checks if a given path exists and if not, creates the needed directories.
            Inputs:
                full_path: path to be checked
    """
    if not os.path.exists(os.path.dirname(full_path)):
        try:
            os.makedirs(os.path.dirname(full_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def readImagesFromPath(image_names):
    """ Takes in a path and a list of image file names to be loaded and returns a list of all loaded images after resize.
           Inputs:
                image_names: list of image names
           Returns:
                List of all loaded and resized images
    """
    returnValue = []
    for image_name in image_names:
        im = Image.open(image_name)
        imArr = np.asarray(im)

        # Remove alpha channel if exists
        if len(imArr.shape) == 3 and imArr.shape[2] == 4:
            if (np.all(imArr[:, :, 3] == imArr[0, 0, 3])):
                imArr = imArr[:, :, 0:3]
        if len(imArr.shape) != 3 or imArr.shape[2] != 3:
            print('Error: Image', image_name, 'is not RGB.')
            sys.exit()

        returnIm = np.asarray(imArr)

        returnValue.append(returnIm)
    return returnValue


def splitTrainValidationAndTestData(all_data_mappings, split_ratio=(0.8, 0.1, 0.1)):
    """Simple function to create train, validation and test splits on the data.
            Inputs:
                all_data_mappings: mappings from the entire dataset
                split_ratio: (train, validation, test) split ratio
            Returns:
                train_data_mappings: mappings for training data
                validation_data_mappings: mappings for validation data
                test_data_mappings: mappings for test data
    """
    if round(sum(split_ratio), 5) != 1.0:
        print("Error: Your splitting ratio should add up to 1")
        sys.exit()

    train_split = int(len(all_data_mappings) * split_ratio[0])
    val_split = train_split + int(len(all_data_mappings) * split_ratio[1])

    train_data_mappings = all_data_mappings[0:train_split]
    validation_data_mappings = all_data_mappings[train_split:val_split]
    test_data_mappings = all_data_mappings[val_split:]

    return [train_data_mappings, validation_data_mappings, test_data_mappings]


def generateDataMapAirSim(folders):
    """ Data map generator for simulator(AirSim) data. Reads label file and returns a list of 'center camera image name - label(s)' tuples
           Inputs:
               folders: list of folders to collect data from
           Returns:
               mappings: All data mappings as a dictionary. Key is the image filepath, the values are a 3-tuple
    """

    all_mappings = {}
    for folder in folders:
        print('Reading data from {0}...'.format(folder))
        current_df = pd.read_csv(os.path.join(folder, 'airsim_rec.txt'), sep='\t')

        for i in range(0, current_df.shape[0], 1):
            steering_label = list(current_df.iloc[i][['Steering']])
            collision_label = list(current_df.iloc[i][['Collision']])
            complexity_label = list(current_df.iloc[i][['Complexity']])

            image_filepath = os.path.join(os.path.join(folder, 'images'), current_df.iloc[i]['ImageFile']).replace('\\',
                                                                                                                   '/')

            # Sanity check
            if (image_filepath in all_mappings):
                print('Error: attempting to add image {0} twice.'.format(image_filepath))

            all_mappings[image_filepath] = (steering_label, collision_label, complexity_label)

    mappings = [(key, all_mappings[key]) for key in all_mappings]

    # random.shuffle(mappings)

    return mappings


def generatorForH5py(data_mappings, chunk_size=32):
    """
    This function batches the data for saving to the H5 file
    """
    for chunk_id in range(0, len(data_mappings), chunk_size):
        # Data is expected to be a dict of <image: (label, previousious_state)>
        # Extract the parts
        data_chunk = data_mappings[chunk_id:chunk_id + chunk_size]
        if len(data_chunk) == chunk_size:
            image_names_chunk = [a for (a, b) in data_chunk]
            steering_label_chunk = np.asarray([b[0] for (a, b) in data_chunk])
            collision_label_chunk = np.asarray([b[1] for (a, b) in data_chunk])
            complexity_label_chunk = np.asarray([b[2] for (a, b) in data_chunk])
            # Flatten and yield as tuple
            yield (image_names_chunk, steering_label_chunk.astype(float), collision_label_chunk.astype(int), complexity_label_chunk.astype(float))
            if chunk_id + chunk_size > len(data_mappings):
                raise StopIteration
    #raise StopIteration


def saveH5pyData(data_mappings, target_file_path):
    """
    Saves H5 data to file
    """
    chunk_size = 32
    gen = generatorForH5py(data_mappings, chunk_size)

    image_names_chunk, steering_label_chunk, collision_label_chunk, complexity_label_chunk = next(gen)
    images_chunk = np.asarray(readImagesFromPath(image_names_chunk))
    row_count = images_chunk.shape[0]

    checkAndCreateDir(target_file_path)
    with h5py.File(target_file_path, 'w') as f:
        # Initialize a resizable dataset to hold the output
        images_chunk_maxshape = (None,) + images_chunk.shape[1:]
        steering_label_chunk_maxshape = (None,) + steering_label_chunk.shape[1:]
        collision_label_chunk_maxshape = (None,) + collision_label_chunk.shape[1:]
        complexity_label_chunk_maxshape = (None,) + complexity_label_chunk.shape[1:]

        dset_images = f.create_dataset('image', shape=images_chunk.shape, maxshape=images_chunk_maxshape,
                                       chunks=images_chunk.shape, dtype=images_chunk.dtype)

        dset_steering = f.create_dataset('steering', shape=steering_label_chunk.shape,
                                               maxshape=steering_label_chunk_maxshape,
                                               chunks=steering_label_chunk.shape, dtype=steering_label_chunk.dtype)

        dset_collision = f.create_dataset('collision', shape=collision_label_chunk.shape, maxshape=collision_label_chunk_maxshape,
                                       chunks=collision_label_chunk.shape, dtype=collision_label_chunk.dtype)

        dset_complexity = f.create_dataset('complexity', shape=complexity_label_chunk.shape, maxshape=complexity_label_chunk_maxshape,
                                       chunks=complexity_label_chunk.shape, dtype=complexity_label_chunk.dtype)


        dset_images[:] = images_chunk
        dset_steering[:] = steering_label_chunk
        dset_collision[:] = collision_label_chunk
        dset_complexity[:] = complexity_label_chunk

        for image_names_chunk, steering_label_chunk, collision_label_chunk, complexity_label_chunk in gen:
            image_chunk = np.asarray(readImagesFromPath(image_names_chunk))

            # Resize the dataset to accommodate the next chunk of rows
            dset_images.resize(row_count + image_chunk.shape[0], axis=0)
            dset_steering.resize(row_count + steering_label_chunk.shape[0], axis=0)
            dset_collision.resize(row_count + collision_label_chunk.shape[0], axis=0)
            dset_complexity.resize(row_count + complexity_label_chunk.shape[0], axis=0)
            # Write the next chunk
            dset_images[row_count:] = image_chunk
            dset_steering[row_count:] = steering_label_chunk
            dset_collision[row_count:] = collision_label_chunk
            dset_complexity[row_count:] = complexity_label_chunk

            # Increment the row count
            row_count += image_chunk.shape[0]


def cook(folders, output_directory, train_eval_test_split):
    """ Primary function for data pre-processing. Reads and saves all data as h5 files.
            Inputs:
                folders: a list of all data folders
                output_directory: location for saving h5 files
                train_eval_test_split: dataset split ratio
    """
    output_files = [os.path.join(output_directory, f) for f in ['train.h5', 'eval.h5', 'test.h5']]
    if (any([os.path.isfile(f) for f in output_files])):
        print("Preprocessed data already exists at: {0}. Skipping preprocessing.".format(output_directory))

    else:
        all_data_mappings = generateDataMapAirSim(folders)

        split_mappings = splitTrainValidationAndTestData(all_data_mappings, split_ratio=train_eval_test_split)
        for i in range(0, len(split_mappings), 1):
            print('Processing {0}...'.format(output_files[i]))
            saveH5pyData(split_mappings[i], output_files[i])
            print('Finished saving {0}.'.format(output_files[i]))