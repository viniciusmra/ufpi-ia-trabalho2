import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    f = sigmoid(x)
    return f * (1 - f)

def initialize_weights_and_biases(layer_sizes):
    weights = []
    biases = []
    for i in range(1, len(layer_sizes)):
        weights.append(np.random.rand(layer_sizes[i-1], layer_sizes[i]))
        biases.append(np.random.rand(layer_sizes[i]))
    return weights, biases

def forward_propagation(inputs, weights, biases):
    activations = [inputs]
    for i in range(len(weights)):
        input_to_layer = np.dot(activations[i], weights[i]) + biases[i]
        activation = sigmoid(input_to_layer)
        activations.append(activation)
    return activations

def backpropagation(inputs, targets, weights, biases, learning_rate):
    num_layers = len(weights)
    activations = forward_propagation(inputs, weights, biases)

    output_error = targets - activations[-1]
    deltas = [output_error * sigmoid_derivative(activations[-1])]

    for i in range(num_layers - 2, -1, -1):
        delta = np.dot(deltas[0], weights[i+1].T) * sigmoid_derivative(activations[i+1])
        deltas.insert(0, delta)

    for i in range(num_layers - 1):
        weights[i] += learning_rate * np.outer(activations[i], deltas[i])
        biases[i] += learning_rate * deltas[i]

    return weights, biases

def train(inputs, targets, layer_sizes, learning_rate, num_epochs):
    weights, biases = initialize_weights_and_biases(layer_sizes)
    for epoch in range(num_epochs):
        for i in range(len(inputs)):
            weights, biases = backpropagation(inputs[i], targets[i], weights, biases, learning_rate)

    return weights, biases

# Exemplo de uso
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [1]])

layer_sizes = [2,2,1]  # Número de neurônios em cada camada
learning_rate = 0.1
num_epochs = 10000

trained_weights, trained_biases = train(inputs, targets, layer_sizes, learning_rate, num_epochs)
print(trained_weights)
print(trained_biases)