import numpy as np
import os
import matplotlib.pyplot as plt


def infer(cnn, classes, Y, path="../cake-classification/images/test"):
    probabilities = []
    max_prob = []
    for class_label, klass in enumerate(classes):
        image_files = os.listdir(path + "/" + klass)
        for imagename in image_files:
            image_path = path + "/" + klass + "/" + imagename
            image = plt.imread(image_path) / 255.0
            # print(image_path)
            prediction, probability = cnn.inference(image[None, :, :, :])
            probabilities.append(probability)
            max_prob.append(probability.max()*100)
    predicted_labels = np.array([0] * 300)
    for i in range(300):
        predicted_labels[i] = np.argmax(probabilities[i])

    test_acc = (predicted_labels == Y).mean()
    print('max prob: ', max_prob)
    print("Test accuracy: {:.2f}%".format(test_acc * 100))
    return test_acc, predicted_labels, max_prob
