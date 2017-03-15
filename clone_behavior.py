import csv
import cv2
import numpy as np
import pickle
import os
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping

#Try Google's Network?
from keras.applications.inception_v3 import InceptionV3

training_samples = '/Users/adelman/Code/sdc/P3-files/driving_log_crop.csv'
training_file = 'training.p'

def read_training_data():
    if (os.path.exists(training_file)):
        with open(training_file, mode='rb') as f:
            training_data = pickle.load(f)
        X_train = training_data['x']
        y_train = training_data['y']
        print('read file.')
    else:
        #file format is [center_img, left_img, right_img,
        # steering_angle, throttle, break, speed]
        steering_angles = []
        images = []
        with open(training_samples) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                image = cv2.imread(line[0])
                angle = float(line[3])

                images.append(image)
                steering_angles.append(angle)
                images.append(np.fliplr(image))
                steering_angles.append(-angle)

        # convert to numpy array
        X_train = np.asarray(images)
        y_train = np.asarray(steering_angles)

        training_data = {}
        training_data['x'] = X_train
        training_data['y'] = y_train
        with open(training_file, mode='wb') as f:
            pickle.dump(training_data, f)
        print('wrote file.')

    return X_train, y_train


def build_model():
    '''
    Build a compiled Keras model to process the input X (track images) and return an output y (steering angle)
    :return: a Keras model
    '''
    # Use David Silver's simple model to get something working
    model = Sequential()

    # Pre-processing
    model.add(Lambda(lambda x : x/255.-.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(16, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model


#Main application
def main():
    X_train, y_train = read_training_data()

    model = build_model()

    #Quality Callback
    cb = EarlyStopping(monitor='val_loss', min_delta=.01, patience=1, verbose=1, mode='auto')

    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, callbacks=[cb], nb_epoch=5)
    model.save('model.h5')

#Python boiler
if __name__ == '__main__':
    main()