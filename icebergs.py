from tensorflow.keras.losses import Reduction, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD


import autokeras as ak

import numpy as np
import os
import cv2
import json
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from keras.utils import plot_model

import numpy as np
import os
import cv2
import json
from os.path import isfile, join
from sklearn.model_selection import train_test_split

from os.path import isfile, join

import pandas as pd
from sklearn.model_selection import train_test_split
import cv2

import os
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils



def from_json(file_path):
    df_train = pd.read_json(file_path)
    Xtrain = get_scaled_imgs(df_train)
    Ytrain = np.array(df_train['is_iceberg'])
    df_train.inc_angle = df_train.inc_angle.replace('na', 0)
    idx_tr = np.where(df_train.inc_angle > 0)
    Ytrain = Ytrain[idx_tr[0]]
    Xtrain = Xtrain[idx_tr[0], ...]
    Ytrain_new = []
    for y in Ytrain:
        new_Y = [0 for _ in range(2)]
        new_Y[y] = 1
        Ytrain_new.append(new_Y)
    # Ytrain = np.array(Ytrain_new)
    Ytrain = np.array(Ytrain)


    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xtrain, Ytrain, random_state=1, train_size=0.75)
    Xtr_more = get_more_images(Xtrain)
    Ytr_more = np.concatenate((Ytrain, Ytrain, Ytrain))

    return Xtr_more, Ytr_more, Xtest, Ytest
    # return Xtrain, Ytrain, Xtest, Ytest

def get_scaled_imgs(df):
    imgs = []

    for i, row in df.iterrows():
        # make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2  # plus since log(x*y) = log(x) + log(y)

        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())
        im = np.dstack((a, b, c))
        im = cv2.resize(im, (75, 75), interpolation = cv2.INTER_AREA)
        imgs.append(im)

    return np.array(imgs)


def get_more_images(imgs):
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []

    for i in range(0, imgs.shape[0]):
        a = imgs[i, :, :, 0]
        b = imgs[i, :, :, 1]
        c = imgs[i, :, :, 2]

        av = cv2.flip(a, 1)
        ah = cv2.flip(a, 0)
        bv = cv2.flip(b, 1)
        bh = cv2.flip(b, 0)
        cv = cv2.flip(c, 1)
        ch = cv2.flip(c, 0)

        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))

    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)

    more_images = np.concatenate((imgs, v, h))

    return more_images



file_path = 'C:/Users/aliev/Documents/GitHub/nas-fedot/IcebergsDataset/train.json'
Xtrain, Ytrain, Xval, Yval = from_json(file_path=file_path)
# Xtrain, Xval = Xtrain / 255.0, Xval / 255.0
# dataset_train, dataset_valid = (Xtrain, Ytrain), (Xval, Yval)



clf = ak.ImageClassifier(
    num_classes=2,
    multi_label=False,
    loss=None,
    metrics=None,
    max_trials=25,
    directory=None,
    objective="val_loss",
    overwrite=True
)

clf.fit(
    Xtrain,
    Ytrain,
    # Split the training data and use the last 15% as validation data.
    validation_split=0.2,
    epochs=5,
)

# Predict with the best model.
predicted_y = clf.predict(Xval)
print(predicted_y)

# Evaluate the best model with testing data.
print(clf.evaluate(Xval, Yval))

# get the best performing model
model = clf.export_model()
# summarize the loaded model
model.summary()
# save the best performing model to file
model.save('model-ice-25.h5')

plot_model(model, "model-ice-25.pdf")