import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

import autokeras as ak

import numpy as np
import os
import cv2
import json
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from keras.utils import plot_model


def load_images(size=120, is_train=True):

    file_path='C:/Users/aliev/Documents/GitHub/nas-fedot/10cls_Generated_dataset'
    with open('C:/Users/aliev/Documents/GitHub/nas-fedot/dataset_files/10_labels.json', 'r') as fp:
        labels_dict = json.load(fp)
    with open('C:/Users/aliev/Documents/GitHub/nas-fedot/dataset_files/10_encoded_labels.json', 'r') as fp:
        encoded_labels = json.load(fp)
    Xarr = []
    Yarr = []
    number_of_classes = 10
    files = [f for f in os.listdir(file_path) if isfile(join(file_path, f))]
    files.sort()
    for filename in files:
        image = cv2.imread(join(file_path, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (size, size))
        Xarr.append(image)
        label_names = labels_dict[filename[:-4]]
        each_file_labels = [0 for _ in range(number_of_classes)]
        for name in label_names:
            num_label = encoded_labels[name]
            # each_file_labels.append(num_label)
            each_file_labels[num_label] = 1
        Yarr.append(each_file_labels)
    Xarr = np.array(Xarr)
    Yarr = np.array(Yarr)
    # Xarr = Xarr.reshape(-1, size, size, 1)

    return Xarr, Yarr


def load_patches():
    Xtrain, Ytrain = load_images(size=120, is_train=True)
    new_Ytrain = []
    for y in Ytrain:
        # ys = np.argmax(y)
        new_Ytrain.append(y)
    new_Ytrain = np.array(new_Ytrain)
    Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, new_Ytrain, random_state=1, train_size=0.8)

    return (Xtrain, Ytrain), (Xval, Yval)


(x_train, y_train), (x_test, y_test) = load_patches()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(y_train.shape)  # (60000,)
print(y_train[:3])  # array([7, 2, 1], dtype=uint8)




clf = ak.ImageClassifier(
    num_classes=10,
    multi_label=False,
    loss=None,
    metrics=None,
    max_trials=20,
    directory=None,
    objective="val_loss",
    overwrite=True
)

clf.fit(
    x_train,
    y_train,
    # Split the training data and use the last 15% as validation data.
    validation_split=0.2,
    epochs=5,
)

# Predict with the best model.
predicted_y = clf.predict(x_test)
print(predicted_y)

# Evaluate the best model with testing data.
print(clf.evaluate(x_test, y_test))

# get the best performing model
model = clf.export_model()
# summarize the loaded model
model.summary()
# save the best performing model to file
model.save('model_autokeras-100.h5')

plot_model(model, "autokeras_model-100.pdf")