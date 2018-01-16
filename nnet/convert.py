import numpy as np

import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape, Permute

def yolo(shape):
    model = Sequential()
    model.add(Convolution2D(16, 3, 3,input_shape=shape,border_mode='same',subsample=(1,1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(64,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(128,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(256,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(512,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(1024,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Convolution2D(1024,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Convolution2D(1024,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dense(4096))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1470))
    return model


# Helper function to load weights from weights-file into YOLO model
def load_weights(model, yolo_weight_file):
    data = np.fromfile(yolo_weight_file, np.float32)
    data = data[4:]

    index = 0
    for layer in model.layers:
        shape = [w.shape for w in layer.get_weights()]
        if shape != []:
            kshape, bshape = shape
            bia = data[index:index + np.prod(bshape)].reshape(bshape)
            index += np.prod(bshape)
            ker = data[index:index + np.prod(kshape)].reshape(kshape)
            index += np.prod(kshape)
            layer.set_weights([ker, bia])


# Load the initial model
keras.backend.set_image_dim_ordering('th')
shape = (3,448,448)
model = yolo(shape)
load_weights(model,'./yolo-tiny.weights')

# Tensorflow backend edition
keras.backend.set_image_dim_ordering('tf')
shape = (448,448,3)
model_full = yolo(shape)

# Transfer weights from Theano model to TensorFlow model_full
for th_layer,tf_layer in zip(model.layers,model_full.layers):
    if th_layer.__class__.__name__ == 'Convolution2D':
        kernel, bias = th_layer.get_weights()
        kernel = np.transpose(kernel,(2,3,1,0))
        tf_layer.set_weights([kernel,bias])
    else:
        tf_layer.set_weights(th_layer.get_weights())


def yoloP1P2P3(shape):
    model = Sequential()
    model.add(Convolution2D(16, 3, 3,input_shape=shape,border_mode='same',subsample=(1,1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(64,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(128,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(256,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(512,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(1024,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Convolution2D(1024,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Convolution2D(1024,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Permute((2,3,1)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dense(4096))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1470))
    return model

model_p1p2p3 = yoloP1P2P3(shape)

# TensorFlow
model_p1p2p3.layers[0].set_weights(model_full.layers[0].get_weights())
model_p1p2p3.layers[3].set_weights(model_full.layers[3].get_weights())
model_p1p2p3.layers[6].set_weights(model_full.layers[6].get_weights())
model_p1p2p3.layers[9].set_weights(model_full.layers[9].get_weights())
model_p1p2p3.layers[12].set_weights(model_full.layers[12].get_weights())
model_p1p2p3.layers[15].set_weights(model_full.layers[15].get_weights())
model_p1p2p3.layers[18].set_weights(model_full.layers[18].get_weights())
model_p1p2p3.layers[20].set_weights(model_full.layers[20].get_weights())
model_p1p2p3.layers[22].set_weights(model_full.layers[22].get_weights())
model_p1p2p3.layers[26].set_weights(model_full.layers[25].get_weights())
model_p1p2p3.layers[27].set_weights(model_full.layers[26].get_weights())
model_p1p2p3.layers[29].set_weights(model_full.layers[28].get_weights())

import coremltools
scale = 2/255.
coreml_model_p1p2p3 = coremltools.converters.keras.convert(model_p1p2p3,
                                                       input_names = 'image',
                                                       output_names = 'output',
                                                       image_input_names = 'image',
                                                       image_scale = scale,
                                                       red_bias = -1.0,
                                                       green_bias = -1.0,
                                                       blue_bias = -1.0)

coreml_model_p1p2p3.author = 'Sri Raghu Malireddi'
coreml_model_p1p2p3.license = 'MIT'
coreml_model_p1p2p3.short_description = 'Yolo - Object Detection'
coreml_model_p1p2p3.input_description['image'] = 'Images from camera in CVPixelBuffer'
coreml_model_p1p2p3.output_description['output'] = 'Output to compute boxes during Post-processing'
coreml_model_p1p2p3.save('TinyYOLOv1.mlmodel')

print "Model Saved!"
