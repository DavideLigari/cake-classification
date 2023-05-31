import numpy as np
import pvml


def train_classifier(trainPath, testPath, epochs=5000, batch_size=50, lr=0.0001, modelName="../models/cake-model-color_histogram"):
    data = np.loadtxt(trainPath)
    X = data[:, :-1]
    Y = data[:, -1].astype(int)

    data = np.loadtxt(testPath)
    Xtest = data[:, :-1]
    Ytest = data[:, -1].astype(int)

    nclasses = Y.max() + 1

    mlp = pvml.MLP([X.shape[1], nclasses])
    train_accs = []
    test_accs = []
    for epoch in range(epochs+1):
        steps = X.shape[0] // batch_size
        mlp.train(X, Y, lr=lr, batch=batch_size, steps=steps)
        if epoch % 100 == 0:
            predictions, probs = mlp.inference(X)
            train_acc = (predictions == Y).mean()
            train_accs.append(train_acc * 100)
            predictions, probs = mlp.inference(Xtest)
            test_acc = (predictions == Ytest).mean()
            test_accs.append(test_acc * 100)

    mlp.save(modelName+".npz")
    return train_accs, test_accs


def train_classifier_v2(X, Y, Xtest, Ytest, epochs=5000, batch_size=50, lr=0.0001, modelName="../models/cake-model-color_histogram"):
    nclasses = Y.max() + 1

    mlp = pvml.MLP([X.shape[1], nclasses])
    train_accs = []
    test_accs = []
    for epoch in range(epochs+1):
        steps = X.shape[0] // batch_size
        mlp.train(X, Y, lr=lr, batch=batch_size, steps=steps)
        if epoch % 100 == 0:
            predictions, probs = mlp.inference(X)
            train_acc = (predictions == Y).mean()
            train_accs.append(train_acc * 100)
            predictions, probs = mlp.inference(Xtest)
            test_acc = (predictions == Ytest).mean()
            test_accs.append(test_acc * 100)

    mlp.save(modelName+".npz")
    return train_accs, test_accs
