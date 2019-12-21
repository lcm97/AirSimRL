from DNNmodel.AdaptedModel import resnet8
from common_flags import FLAGS
import numpy as np
import sys
import os
import gflags
import h5py
from keras import optimizers

def getModel(img_width, img_height, img_channel, weight_path):
    model = resnet8(img_width, img_height, img_channel)

    if weight_path:
        try:
            model.load_weights(weight_path)
            print("Loaded model from {}".format(weight_path))
        except:
            print("Impossible to find weight path. Returning untrained model")

    return model


def train():
    # Create the experiment rootdir if not already there
    if not os.path.exists(FLAGS.experiment_rootdir):
        os.makedirs(FLAGS.experiment_rootdir)

    # Image mode
    if FLAGS.img_mode=='rgb':
        img_channels = 3
    elif FLAGS.img_mode == 'grayscale':
        img_channels = 1
    else:
        raise IOError("Unidentified image mode: use 'grayscale' or 'rgb'")

    img = np.zeros([320, 320, 1], np.uint8)
    #model = getModel(img.shape[0], img.shape[1], img_channels, None)
    model = getModel(None, None, img_channels, None)
    model.summary()

    optimizer = optimizers.Adam(decay=1e-5)
    model.compile(loss=['mean_squared_error','binary_crossentropy','mean_squared_error'],
                  optimizer=optimizer, loss_weights=[1.5, 1, 0.001])
    model.fit([[img]],[[0.5],[0.],[25.]])
    output = model.predict([[img]])
    print(output)
    img = np.zeros([256, 256, 1], np.uint8)
    output = model.predict([[img]])
    model.fit([[img]],[[0.5],[0.],[25.]])
    print(output)





def main(argv):
    # Utility main to load flags
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
      sys.exit(1)
    train()

if __name__ == "__main__":
    main(sys.argv)