from DNNmodel.AdaptedModel import resnet8
from common_flags import FLAGS
import numpy as np
import sys
import os
import gflags
import h5py
from keras import optimizers
from CookData.Generator import DriveDataGenerator
import h5py
from PIL import ImageDraw
import math
import matplotlib.pyplot as plt
import keras.backend as K
from keras.preprocessing import image
from keras.callbacks import EarlyStopping,TensorBoard, History, ReduceLROnPlateau,CSVLogger

def draw_image_with_label(img, label, prediction=None):
    theta = label * 0.69  # Steering range for the car is +- 40 degrees -> 0.69 radians
    line_length = 50
    line_thickness = 3
    label_line_color = (255, 0, 0)
    prediction_line_color = (0, 0, 255)
    pil_image = image.array_to_img(img, K.image_data_format(), scale=True)
    print('Actual Steering Angle = {0}'.format(label))
    draw_image = pil_image.copy()
    image_draw = ImageDraw.Draw(draw_image)
    first_point = (int(img.shape[1] / 2), img.shape[0])
    second_point = (
    int((img.shape[1] / 2) + (line_length * math.sin(theta))), int(img.shape[0] - (line_length * math.cos(theta))))
    image_draw.line([first_point, second_point], fill=label_line_color, width=line_thickness)

    if (prediction is not None):
        print('Predicted Steering Angle = {0}'.format(prediction))
        print('L1 Error: {0}'.format(abs(prediction - label)))
        theta = prediction * 0.69
        second_point = (
        int((img.shape[1] / 2) + (line_length * math.sin(theta))), int(img.shape[0] - (line_length * math.cos(theta))))
        image_draw.line([first_point, second_point], fill=prediction_line_color, width=line_thickness)

    del image_draw
    plt.imshow(draw_image)
    plt.show()

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

    train_dataset = h5py.File(os.path.join(FLAGS.cooked_data_dir, 'train.h5'), 'r')
    eval_dataset = h5py.File(os.path.join(FLAGS.cooked_data_dir, 'eval.h5'), 'r')
    test_dataset = h5py.File(os.path.join(FLAGS.cooked_data_dir, 'test.h5'), 'r')
    num_train_examples = train_dataset['image'].shape[0]
    num_eval_examples = eval_dataset['image'].shape[0]
    num_test_examples = test_dataset['image'].shape[0]

    batch_size = 32
    data_generator = DriveDataGenerator(rescale=1. / 255., horizontal_flip=True, brightness_range=[0.8, 1.2])

    train_generator = data_generator.flow \
        (train_dataset['image'], train_dataset['steering'], train_dataset['collision'], train_dataset['complexity'],
         batch_size=batch_size, zero_drop_percentage=0.1)
    eval_generator = data_generator.flow \
        (eval_dataset['image'], eval_dataset['steering'], eval_dataset['collision'], eval_dataset['complexity'],
         batch_size=batch_size, zero_drop_percentage=0.1)

    [sample_batch_train_data, sample_batch_label] = next(train_generator)
    for i in range(0, 3, 1):
        draw_image_with_label(sample_batch_train_data[i], sample_batch_label[1][i])


    model = getModel(None, None, img_channels, None)
    model.summary()

    optimizer = optimizers.Adam(decay=1e-5)
    model.compile(loss=['mean_squared_error','binary_crossentropy','mean_squared_error'],
                  optimizer=optimizer, loss_weights=[1.5, 1, 0.01])

    # callbacks
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    history = History()

    model.fit_generator(train_generator, steps_per_epoch=num_train_examples // batch_size, epochs=10,
                                  callbacks=[lr_reducer, history],validation_data=eval_generator, validation_steps=num_eval_examples // batch_size,
                                  verbose=2)
    model.save('./model/model.h5')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


    # model.fit([[img]],[[0.5],[0.],[25.]])
    # output = model.predict([[img]])
    # print(output)
    # img = np.zeros([256, 256, 1], np.uint8)
    # output = model.predict([[img]])
    # model.fit([[img]],[[0.5],[0.],[25.]])
    # print(output)



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