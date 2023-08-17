import numpy as np
import network
import sys

import matplotlib.pyplot as plt

class Training:
    def __init__(self, all_training_data, network, learning_rate):
        self.network = network
        self.all_training_data = all_training_data
        self.outputs = []
        self.error = 0
        self.error_derivative = 0
        self.learning_rate = learning_rate


    def training_with_threshold(self, threshold):
        epoch = 0
        last_error = 0
        sum_error = 0
        errors = []
        weights_list = []
        
        for data in self.all_training_data:
            print(f"> > > Epoch: {epoch}")
            epoch += 1
            activations, error, weights, biases = self.training_session(data)
            sum_error += error
            sum_error/len(self.all_training_data)
        self.network.show()
        print(f"Error: {error}\n")

        while True:#(last_error - sum_error)/sum_error > threshold):
            print(f"> > > Epoch: {epoch}")
            epoch += 1
            last_error = sum_error
            sum_error = 0
            for data in self.all_training_data:
                activations, error, weights, biases = self.training_session(data)
                sum_error += error
            weights = []
            for layer in self.network.layers:
                for perceptron in layer.perceptrons:
                    for weight in perceptron.weights:
                        weights.append(weight)
            weights_list.append(weights)
            sum_error = sum_error/len(self.all_training_data)
            errors.append(sum_error)
            self.network.show()
            print(f"Error: {error}\n")
            if((abs(last_error - sum_error)/sum_error) < threshold):
                return errors, weights_list
        
    def training_with_limit(self, limit, full_print):
        weights_list = []
        errors = []
        for epoch in range(limit):#(last_error - sum_error)/sum_error > threshold):
            sum_error = 0
            print(f"> > > Epoch: {epoch}")
            for index, data in enumerate(self.all_training_data):
                activations, error, weights, biases = self.training_session(data)
                if(full_print):
                    print(f"Data: {index} - {data} - {activations[-1]}")
                    self.network.show()
                    print(f"Error: {error}\n")

                sum_error += error
            weights = []
            for layer in self.network.layers:
                for perceptron in layer.perceptrons:
                    for weight in perceptron.weights:
                        weights.append(weight)
            weights_list.append(weights)
            sum_error = sum_error/len(self.all_training_data)
            errors.append(sum_error)
            if(not(full_print)):
                self.network.show()
                print(f"Error: {error}\n")
    
        return errors, weights_list

    def training_session(self,data):
        inputs, target_outputs = data
        number_layers = len(self.network.layers)
        activations = self.network.forward(inputs)
        output = activations[-1]
        error = (1/2)*(activations[-1][0] - target_outputs[0])**2
        error_derivative = (output - target_outputs)

        deltas = [error_derivative * network.sigmoid_derivative(output)]
        # CALCULO DOS DELTAS
        for i in range(number_layers - 2, -1, -1):
            delta = np.dot(deltas[0], self.network.weights[i+1].T) * network.sigmoid_derivative(activations[i])
            deltas.insert(0, delta)
        
        # ATUALIZAÇÃO DOS PESOS

        for i in range(number_layers):
            if(i == 0):
                adjust_values = (self.learning_rate * np.outer(deltas[i], inputs)).ravel().tolist()

            else:
                adjust_values = (self.learning_rate * np.outer(deltas[i], activations[i-1])).ravel().tolist()

            for index, weight in enumerate(self.network.weights[i]):
                self.network.weights[i][index] = weight - adjust_values[index]
            self.network.biases[i] -= self.learning_rate * deltas[i]

        self.network.put_weights()
        self.network.put_biases()

        
        return activations, error, self.network.weights, self.network.biases
        #print(f"error: {error}")
