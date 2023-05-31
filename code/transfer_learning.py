import pvml


def transfer_learning(mlpPath, cnnPath='../models/pvmlnet.npz'):
    cnn = pvml.CNN.load(cnnPath)
    mlp = pvml.MLP.load(mlpPath)

    cnn.weights[-1] = mlp.weights[0][None, None, :, :]
    cnn.biases[-1] = mlp.biases[0]

    cnn.save("../models/cakes-cnn_transferred.npz")
    return cnn
