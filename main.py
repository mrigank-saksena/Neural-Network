import mlp as mlp

# Main program to take user input for training and testing

option = ""
continueprogram = ""
# User Input
while option != "0" and option != "1":
    option = input('Would you like to train or test? (0 for train, 1 for test) ')
    if option != '0' and option != '1':
        print('Invalid input!')

# Train
if option == '0':
    initial = input('Enter the initial neural network file: ')
    inputNodes, hiddenNodes, outputNodes, weights = mlp.readnn(initial)

    trainingFile = input('Enter the neural network training data file: ')
    trainingData = mlp.readdata(trainingFile)

    learn = float(input('Enter the learning rate: '))
    epochs = int(input('Enter the number of epochs: '))
    print("Training in progress!")
    weights = mlp.train(weights, trainingData, learn, epochs)

    print("Training completed!")
    trained = input('Enter the output file for the trained neural network: ')
    mlp.writetrained(trained, inputNodes, hiddenNodes, outputNodes, weights)
    continueprogram = input('Would you like to test this neural network? (Y/N) ')

    if continueprogram.upper() == 'Y':

        inputNodes, hiddenNodes, outputNodes, weights = mlp.readnn(trained)
        testing = input('Enter the neural network testing data file: ')
        trainingData = mlp.readdata(testing)

        confusion = mlp.test(outputNodes, weights, trainingData)

        results = input('Enter the output file for the neural network results: ')
        mlp.writeresults(results, outputNodes, confusion)

# Test
else:
    trained = input('Enter the trained neural network file: ')
    inputNodes, hiddenNodes, outputNodes, weights = mlp.readnn(trained)

    testing = input('Enter the neural network testing data file: ')
    trainingData = mlp.readdata(testing)

    confusion = mlp.test(outputNodes, weights, trainingData)

    results = input('Enter the output file for the neural network results: ')
    mlp.writeresults(results, outputNodes, confusion)