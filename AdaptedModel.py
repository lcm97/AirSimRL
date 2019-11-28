from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from SpatialPyramidPooling import SpatialPyramidPooling
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

batch_size = 64

def resnet8(img_width, img_height, num_channels):
    """
    Define model architecture.

    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.

    # Returns
       model: A Model instance.
    """
    # Input
    img_input = Input(shape=(img_width, img_height, num_channels))

    x1 = Conv2D(32, (5, 5), strides=[2, 2], padding='same')(img_input)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2, 2])(x1)

    # First residual block
    x2 = BatchNormalization()(x1)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), strides=[2, 2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x2)

    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x2)

    x1 = Conv2D(32, (1, 1), strides=[2, 2], padding='same')(x1)
    x3 = add([x1, x2])

    # Second residual block
    x4 = BatchNormalization()(x3)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), strides=[2, 2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

    x3 = Conv2D(64, (1, 1), strides=[2, 2], padding='same')(x3)
    x5 = add([x3, x4])

    # Third residual block
    x6 = BatchNormalization()(x5)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), strides=[2, 2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x6)

    x6 = BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x6)

    x5 = Conv2D(128, (1, 1), strides=[2, 2], padding='same')(x5)
    x7 = add([x5, x6])

    x7 = SpatialPyramidPooling([1, 2, 4])(x7)
    x = Activation('relu')(x7)
    x = Dropout(0.5)(x)

    # Steering channel
    steering = Dense(1,name="steering")(x)

    # Collision channel
    collision = Dense(2,name='collision')(x)
    collision = Activation('softmax')(collision)

    # Complexity channel
    complexity = Dense(1,name='complexity')(x)

    model = Model(inputs=[img_input], outputs=[steering , collision , complexity])
    return model



# model = resnet8(None,None,3)
# model.alpha = tf.Variable(1, trainable=False, name='alpha', dtype=tf.float32)
# model.beta = tf.Variable(0.2, trainable=False, name='beta', dtype=tf.float32)
# optimizer = optimizers.Adam(decay=1e-5)
# model.compile(loss=[losses.mean_squared_error,
#                         losses.binary_crossentropy,
#                         losses.mean_squared_error],
#                         optimizer=optimizer,
#                         loss_weights=[model.alpha, model.beta, model.alpha],
#                         metrics=['acc','acc','acc'])
#
# num_channels = 3
# model.fit(np.random.rand(batch_size, 64, 64, num_channels), [np.zeros((batch_size, 1)), np.zeros((batch_size, 2)),np.zeros((batch_size, 1)) ])
# model.fit(np.random.rand(batch_size, 32, 32, num_channels), [np.zeros((batch_size, 1)), np.zeros((batch_size, 2)),np.zeros((batch_size, 1)) ])
#
# Y = model.predict(np.random.rand(1, 64, 64, num_channels))
# print(Y)
# Y = model.predict(np.random.rand(1, 32, 32, num_channels))
# print(Y)
# Y = model.predict(np.random.rand(1, 224, 224, num_channels))