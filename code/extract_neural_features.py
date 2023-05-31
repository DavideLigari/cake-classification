import numpy as np
import pvml
import matplotlib.pyplot as plt
import os


def get_classes(path):
    classes = os.listdir(path)
    classes.sort()
    return classes


def extract_neural_features(im, net, activation_layer):
    activations = net.forward(im[None, :, :, :])
    features = activations[activation_layer]
    features = features.reshape(-1)
    return features


def process_directory(path, net, classes, activation_layer=-1):
    all_features = []
    all_labels = []
    for klass_label, klass in enumerate(classes):
        image_files = os.listdir(path + "/" + klass)
        for imagename in image_files:
            image_path = path + "/" + klass + "/" + imagename
            image = plt.imread(image_path) / 255.0
            features = extract_neural_features(image, net, activation_layer)
            all_features.append(features)
            all_labels.append(klass_label)
    X = np.stack(all_features, 0)
    Y = np.array(all_labels)
    return X, Y
