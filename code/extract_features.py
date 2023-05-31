import numpy as np
import matplotlib.pyplot as plt
import os
import image_features


def get_classes(path):
    classes = os.listdir(path)
    classes.sort()
    return classes


def process_directory(path, classes, feature_elab=image_features.color_histogram):
    all_features = []
    all_labels = []
    klass_label = 0
    for klass in classes:
        image_files = os.listdir(path + "/" + klass)
        for imagename in image_files:
            image_path = path + "/" + klass + "/" + imagename
            image = plt.imread(image_path) / 255.0
            features = feature_elab(image)
            # features = image_features.edge_direction_histogram(image)
            # features = image_features.cooccurrence_matrix(image)
            features = features.reshape(-1)
            all_features.append(features)
            all_labels.append(klass_label)
        klass_label += 1
    X = np.stack(all_features, 0)
    Y = np.array(all_labels)
    return X, Y


def get_images_as_features(path, classes):
    all_features = []
    all_labels = []
    klass_label = 0
    for klass in classes:
        image_files = os.listdir(path + "/" + klass)
        for imagename in image_files:
            image_path = path + "/" + klass + "/" + imagename
            image = plt.imread(image_path) / 255.0
            # features = image_features.edge_direction_histogram(image)
            # features = image_features.cooccurrence_matrix(image)
            image = image.reshape(-1)
            all_features.append(image)
            all_labels.append(klass_label)
        klass_label += 1
    X = np.stack(all_features, 0)
    Y = np.array(all_labels)
    return X, Y
