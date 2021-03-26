from __future__ import print_function
import sys
import yaml
import os
import collections
import pickle
import os

import keras

def train(model, data, settings):
    print("-- RUNNING TRAINING --", flush=True)



    with open(os.path.join(data, 'trainx.pyp'), 'rb') as fh:
        x_train = pickle.loads(fh.read())
    with open(os.path.join(data, 'trainy.pyp'),'rb') as fh:
        y_train = pickle.loads(fh.read())

    print("x_train shape: : ", x_train.shape)

    model.fit(x_train, y_train, batch_size=settings['batch_size'], epochs=settings['epochs'], verbose=1)

    print("-- TRAINING COMPLETED --", flush=True)
    return model



if __name__ == '__main__':

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise(e)

    from models.keras_models.vgg import VGG

    from fedn.utils.kerasweights import KerasWeightsHelper
    import tensorflow as tf
    gpu = len(tf.config.list_physical_devices('GPU')) > 0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")

    helper = KerasWeightsHelper()
    weights = helper.load_model(sys.argv[1])
    model = VGG(dimension='VGG11')
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    model.set_weights(weights)
    print("list /app/data")
    import os

    arr = os.listdir('/app/data')
    print(arr)
    model = train(model, '/app/data', settings)
    helper.save_model(model.get_weights(), sys.argv[2])


