import numpy as np
import json

class Network:
    def __init__(self, activation_function):
        self.layers = []
        self.number_inputs = 0
        self.activation_function = activation_function

    def load(self, filepath):
        with open(filepath, 'r') as file:
            load_file = json.load(file)
        self.layers = []
        self.number_inputs = load_file["number_inputs"]
        layer_inputs = self.number_inputs
        for layer in load_file["layers"]:
            self.layers.append(Layer(self.activation_function, layer_inputs, layer["perceptrons"]))
            layer_inputs = layer["number_perceptrons"]
    
    def create_random(self, number_inputs, layers):
        self.number_inputs = number_inputs
        layer_inputs = self.number_inputs
        for num_perceptrons in layers:
            layer = Layer(self.activation_function, layer_inputs)
            for _ in range(num_perceptrons):
                layer.add_perceptron(self.activation_function, layer_inputs)
            self.layers.append(layer)
            layer_inputs = num_perceptrons
    
    def save(self, filepath):
        data = {}
        data["number_inputs"] = self.number_inputs
        data["layers"] = []
        for layer in self.layers:
            layer_json = {"number_perceptrons": len(layer.perceptrons)}
            layer_json["perceptrons"] = []
            for perceptron in layer.perceptrons:
                layer_json["perceptrons"].append({"weights": perceptron.weights.tolist(), "bias": perceptron.bias})
            data["layers"].append(layer_json)

        with open(filepath, 'w') as arquivo:
            json.dump(data, arquivo)

class Layer:
    """
    A classe Layer representa uma camada em uma rede neural. Uma camada em uma rede neural é um 
    conjunto de neurônios que processam os dados de entrada e geram saídas. Cada neurônio em uma 
    camada recebe entradas ponderadas, aplica uma função de ativação aos valores ponderados e produz uma saída. 
    A classe Layer agrupa um conjunto de objetos Perceptron, que são os neurônios individuais que compõem a camada.
    """
    def __init__(self, activation_function, number_inputs, perceptrons=None):
        self.perceptrons = []
        self.activation_function = activation_function
        self.number_inputs = number_inputs
        if perceptrons is None:
            self.perceptrons = []
        else:
            for perceptron in perceptrons:
                self.perceptrons.append(Perceptron(activation_function, number_inputs, perceptron["weights"], perceptron["bias"]))
    
    def add_perceptron(self, activation_function, number_inputs):
        perceptron = Perceptron(activation_function, number_inputs)
        self.perceptrons.append(perceptron)
    
    def show(self):
        for index, perceptron in enumerate(self.perceptrons):
            print(f"Perceptron {index+1}: ", end='')
            perceptron.show()

class Perceptron:
    def __init__(self, activation_function, number_inputs, weights=None, bias=None):
        self.number_inputs = number_inputs
        self.activation_function = activation_function
        if weights is None:
            self.weights = np.random.rand(number_inputs)
        else:
            self.weights = weights
        if bias is None:
            self.bias = np.random.rand()
        else:
            self.bias = bias

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        output = self.activation_function(weighted_sum)
        return output
    
    def show(self):
        print(f"weights: {self.weights} - bias: {self.bias}")

def step_function(x):
    return 1 if x >= 0 else 0

if __name__ == "__main__":
    net = Network(activation_function=step_function)

    # Criação da rede camada por camada e neurônio por neurônio
    net.create_random(2, [2, 3, 1]) 

    for idx, layer in enumerate(net.layers):
        print(f"Layer {idx + 1}:")
        layer.show()

    # Dados de treinamento
    training_data = [
        (np.array([0, 0]), 0),
        (np.array([0, 1]), 1),
        (np.array([1, 0]), 1),
        (np.array([1, 1]), 0)
    ]

# learning_rate = 0.1

# # Crie sua rede neural e camadas aqui (como no código original)

# desired_error = 0.01  # Define your desired error threshold
# total_error = float('inf')  # Initialize total_error with a large value

# epoch = 0
# while total_error > desired_error:
#     total_error = 0

#     print(f"\nEpoch {epoch + 1}:")

#     for inputs, target in training_data:
#         # Forward pass
#         current_output = inputs
#         for layer in net.layers:
#             current_output = np.array([perceptron.predict(current_output) for perceptron in layer.perceptrons])

#         # Calculate error
#         error = target - current_output
#         total_error += np.sum(np.abs(error))

#         # Backpropagation and weight update
#         for layer in reversed(net.layers):
#             for perceptron, perceptron_error in zip(layer.perceptrons, error):
#                 delta = perceptron_error * learning_rate
#                 perceptron.weights += delta * current_output
#                 perceptron.bias += delta

#                 print(f"Updated weights: {perceptron.weights} - Updated bias: {perceptron.bias}")

#     print(f"Total Error: {total_error}")
#     epoch += 1

# # Mostrar os pesos e bias finais após o treinamento
# print("\nFinal Weights and Biases:")
# for idx, layer in enumerate(net.layers):
#     for perceptron_idx, perceptron in enumerate(layer.perceptrons):
#         print(f"Layer {idx + 1}, Perceptron {perceptron_idx + 1}:")
#         perceptron.show()

# net.save("./networks/network3.json")


    num_epochs = 10
    learning_rate = 0.1

    for epoch in range(num_epochs):
        total_error = 0

        print(f"\nEpoch {epoch + 1}:")

        for inputs, target in training_data:
            print(f"Input: {inputs}, Target: {target}")

            # Forward pass
            current_output = inputs
            for layer_idx, layer in enumerate(net.layers):
                print(f"\nLayer {layer_idx + 1}:")

                current_output = np.array([perceptron.predict(current_output) for perceptron in layer.perceptrons])

                for perceptron_idx, perceptron in enumerate(layer.perceptrons):
                    print(f"  Perceptron {perceptron_idx + 1}:")
                    print(f"    Weights: {perceptron.weights}")
                    print(f"    Bias: {perceptron.bias}")
                    print(f"    Output: {current_output[perceptron_idx]}")

            # Calculate error
            error = target - current_output
            total_error += np.sum(np.abs(error))

            # Backpropagation and weight update
            for layer_idx, layer in reversed(list(enumerate(net.layers))):
                print(f"\nBackpropagation - Layer {layer_idx + 1}:")

                for perceptron_idx, (perceptron, perceptron_error) in enumerate(zip(layer.perceptrons, error)):
                    delta = perceptron_error * learning_rate
                    perceptron.weights += delta * current_output
                    perceptron.bias += delta

                    print(f"  Perceptron {perceptron_idx + 1}:")
                    print(f"    Updated Weights: {perceptron.weights}")
                    print(f"    Updated Bias: {perceptron.bias}")

        print(f"Total Error: {total_error}")
    
    # Mostrar os pesos e bias finais após o treinamento
    print("\nFinal Weights and Biases:")
    for idx, layer in enumerate(net.layers):
        for perceptron_idx, perceptron in enumerate(layer.perceptrons):
            print(f"Layer {idx + 1}, Perceptron {perceptron_idx + 1}:")
            perceptron.show()

    net.save("./networks/network3.json")
