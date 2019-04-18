import sys
import os
# import plaidml.keras
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# plaidml.keras.install_backend()

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
import time

def main():
    start = time.time()




    train_data_path = 'data/train'
    validation_data_path = 'data/val'
    save_path = "data/new_pics"

    """
    Parameters
    """
    epochs = 15
    img_width, img_height = 150, 150
    batch_size = 100
    samples_per_epoch = 1000
    validation_steps = 300
    nb_filters1 = 32
    nb_filters2 = 64
    conv1_size = 3
    conv2_size = 2
    pool_size = 2
    classes_num = 47
    lr = 0.0004

    model = Sequential()
    model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, border_mode ="same", input_shape=(img_width, img_height, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, border_mode ="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), dim_ordering='th'))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(classes_num, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=lr),
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.3,height_shift_range=0.3,brightness_range=[0.3,1.0],
                                   zoom_range=[0.5,1.2],channel_shift_range=5,rescale=1/255)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_path,
        save_format="png", save_to_dir=save_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    """
    Tensorboard log
    """
    log_dir = './tf-log/'
    tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
    cbks = [tb_cb]

    model.fit_generator(
        train_generator,
        samples_per_epoch=samples_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=cbks,
        validation_steps=validation_steps)

    target_dir = './models/'
    if not os.path.exists(target_dir):
      os.mkdir(target_dir)
    model.save('./models/model.h5')
    model.save_weights('./models/weights.h5')

    #Calculate execution time
    end = time.time()
    dur = end-start

if __name__=="__main__":
    main()


