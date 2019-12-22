import pandas as pd
import os
import sys
from common_flags import FLAGS
import gflags
import CookData.Cooking as Cooking

def cook():
    #data_folders = ['320','64','128','256','32']
    data_folders = ['256']
    full_path_raw_folders = [os.path.join(FLAGS.raw_data_dir, f) for f in data_folders]
    dataframes = []
    for folder in full_path_raw_folders:
        current_dataframe = pd.read_csv(os.path.join(folder, 'airsim_rec.txt'), sep='\t')
        current_dataframe['Folder'] = folder
        dataframes.append(current_dataframe)
    dataset = pd.concat(dataframes, axis=0, sort=True)
    print('Number of data points: {0}'.format(dataset.shape[0]))
    train_eval_test_split = [0.8, 0.1, 0.1]
    Cooking.cook(full_path_raw_folders, FLAGS.cooked_data_dir, train_eval_test_split)


def main(argv):
    # Utility main to load flags
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
      sys.exit(1)
    cook()

if __name__ == "__main__":
    main(sys.argv)