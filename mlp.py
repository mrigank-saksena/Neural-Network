from math import trunc
import numpy as np


# Neural Network Implementation


# Reads the inputted neural network
# Parameters: Neural Network file name
# Returns: Number of nodes per layer and weights
def readnn(filename):
    input = open(filename, 'r')
    line = input.readline()
    nodes = []
    for val in line.split():
        nodes.append(int(val))

    # Reading in nodes per layer
    inputnodes = nodes[0]
    hiddennodes = nodes[1]
    outputnodes = nodes[2]

    # Read weights from input to hidden nodes
    inweights = np.zeros([hiddennodes, inputnodes + 1])
    for i in range(hiddennodes):
        line = input.readline()
        vals = []
        for val in line.split():
            vals.append(float(val))
        inweights[i, :] = vals

    # Read weights from hidden to output nodes
    outweights = np.zeros([outputnodes, hiddennodes + 1])
    for i in range(outputnodes):
        line = input.readline()
        vals = []
        for val in line.split():
            vals.append(float(val))
        outweights[i, :] = vals

    return inputnodes, hiddennodes, outputnodes, [inweights, outweights]


# Reads the inputted training/testing data
# Parameters: Data file (training or testing)
# Returns: Mapped list of inputs to outputs
def readdata(inputdata):
    data = open(inputdata, 'r')
    line = data.readline()
    vals = []
    for val in line.split():
        vals.append(int(val))

    # Reading in count of observations, inputs, and outputs
    obs = vals[0]
    inp = vals[1]
    out = vals[2]
    inputs = np.zeros([obs, inp])
    outputs = np.zeros([obs, out])

    for i in range(obs):
        line = data.readline()
        vals = []
        for val in line.split():
            vals.append(float(val))
        inputs[i, :] = vals[:inp]
        outputs[i, :] = vals[inp:]

    # Mapping inputs to outputs
    # https://python-reference.readthedocs.io/en/latest/docs/functions/zip.html
    mapped = list(zip(inputs, outputs))
    return mapped


# Trains the neural network
# Parameters: Weights, data (returned from readdata), learning rate, and epochs
# Returns: New trained weights of neural network
def train(weights, data, rate, epochs):
    for i in range(epochs):
        for observation, classification in data:
            updates = []
            for weight in weights:
                updates.append(np.zeros(weight.shape))

            # Propagate the inputs through neural network
            activation = observation
            activations = [observation]
            siginputs = []

            for weight in weights:
                input = np.dot(weight, np.insert(activation, 0, -1))
                siginputs.append(input)
                activation = sig(input)
                activations.append(activation)

            # Back-propagation
            # Help from https://dev.to/shamdasani/build-a-flexible-neural-network-with-backpropagation-in-python
            delta = sigprime(siginputs[len(siginputs) - 1]) * (classification - activations[len(activations) - 1])
            updates[len(updates) - 1] = np.insert(np.outer(delta, activations[1]), 0, -delta, axis=1)

            delta = sigprime(siginputs[len(siginputs) - 2]) * (np.matmul(weights[len(weights) - 1][:, 1:].T, delta))
            updates[len(updates) - 2] = np.insert(np.outer(delta, activations[0]), 0, -delta, axis=1)

            # Updates weights for trained neural network
            weights = [(weight + rate * update) for weight, update in zip(weights, updates)]

    return weights


# Prints the trained neural network
# Parameters: Output file name, number of input, hidden, and output nodes, weights of trained neural network
def writetrained(filename, inputnodes, hiddennodes, outputnodes, weights):
    outfile = open(filename, 'w')
    # Prints number of nodes per layer on the first line
    print(' '.join(str(n) for n in [inputnodes, hiddennodes, outputnodes]), file=outfile)
    # Prints the weights entering the hidden layer
    for i in range(hiddennodes):
        print(' '.join('{:.3f}'.format(j) for j in weights[0][i, :]), file=outfile)
    # Prints the weights entering the output layer
    for i in range(outputnodes):
        print(' '.join('{:.3f}'.format(j) for j in weights[1][i, :]), file=outfile)
    outfile.close()


# Tests the neural network
# Parameters: number of output nodes, weights, and testing data
# Returns: The confusion matrix
def test(outputnodes, weights, data):
    confusion = np.zeros([outputnodes, 2, 2])

    for inputs, outputs in data:
        predicted = np.floor(forwardpropogate(inputs, weights) + 0.5)  # Rounded up

        # Filling in confusion matrix
        # https://www.python-course.eu/confusion_matrix.php
        for i in range(outputnodes):
            if predicted[i] == 1 and outputs[i] == 1:
                confusion[i, 0, 0] += 1
            elif predicted[i] == 1 and outputs[i] == 0:
                confusion[i, 0, 1] += 1
            elif predicted[i] == 0 and outputs[i] == 1:
                confusion[i, 1, 0] += 1
            elif predicted[i] == 0 and outputs[i] == 0:
                confusion[i, 1, 1] += 1

    return confusion


# Prints the results of the neural network
# Parameters: output file, number of output nodes, and the confusion matrix (returned from test)
def writeresults(filename, outputs, confusion):
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    output = open(filename, 'w')

    # Prints the classification metrics for each class
    for i in range(outputs):
        accuracies.append((confusion[i, 0, 0] + confusion[i, 1, 1]) / confusion[i].sum())
        precisions.append(confusion[i, 0, 0] / confusion[i, 0, :].sum())
        recalls.append(confusion[i, 0, 0] / confusion[i, :, 0].sum())
        f1s.append(2 * precisions[len(precisions) - 1] * recalls[len(recalls) - 1] / (
                    precisions[len(precisions) - 1] + recalls[len(recalls) - 1]))

        print(' '.join('{:d}'.format(trunc(j)) for j in confusion[i].flatten()), end=' ', file=output)
        print(' '.join('{:.3f}'.format(j) for j in [accuracies[len(accuracies) - 1], precisions[len(precisions) - 1],
                                 recalls[len(recalls) - 1], f1s[len(f1s) - 1]]), file=output)

    # Micro-averaging of the metrics
    microconfusion = confusion.sum(axis=0)
    microaccuracy = (microconfusion[0, 0] + microconfusion[1, 1]) / microconfusion.sum()
    microprecision = microconfusion[0, 0] / microconfusion[0, :].sum()
    microrecall = microconfusion[0, 0] / microconfusion[:, 0].sum()
    microf1 = (2 * microprecision * microrecall) / (microprecision + microrecall)

    print(' '.join('{:.3f}'.format(j) for j in [microaccuracy, microprecision, microrecall, microf1]), file=output)

    # Macro-averaging of the metrics
    macroaccuracy = sum(accuracies) / len(accuracies)
    macroprecision = sum(precisions) / len(precisions)
    macrorecall = sum(recalls) / len(recalls)
    macrof1 = (2 * macroprecision * macrorecall) / (macroprecision + macrorecall)

    print(' '.join('{:.3f}'.format(j) for j in [macroaccuracy, macroprecision, macrorecall, macrof1]), file=output)


# Propagates the input data forward through the neural network
# Parameters: input values and weights
# Returns: values with inputs are propagated through the neural network
def forwardpropogate(inputs, weights):
    results = inputs
    for weight in weights:
        results = np.insert(results, 0, -1)
        results = sig(np.dot(weight, results))
    return results


# Sigmoid function
# Parameters: value to be passed through sigmoid
# Returns: Sigmoid of value passed in: 1/1+e^(-x)
def sig(val):
    return 1.0 / (1.0 + np.exp(-val))


# Derivative of sigmoid function
# Parameters: value to be passed through sigmoid prime
# Returns: Derivative of sigmoid of value passed in: 1/1+e^(-x) * (1 - 1/1+e^(-x))
def sigprime(val):
    return sig(val) * (1 - sig(val))
