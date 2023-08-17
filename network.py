import numpy as np
import json

# Classe network
class Network:
    def __init__(self):
        self.layers = []
        self.number_inputs = 0
        self.activation_function = sigmoid_function
        self.weights = []
        self.biases = []
        self.error = 0

    # Método pra carregar uma rede apartir de um arquivo
    def load(self, filepath):
        with open(filepath, 'r') as file:
            load_file = json.load(file)
        self.layers = []
        self.number_inputs = load_file["number_inputs"]
        layer_inputs = self.number_inputs
        for layer in load_file["layers"]:
            self.layers.append(Layer(self.activation_function, layer_inputs, layer["perceptrons"]))
            layer_inputs = layer["number_perceptrons"]
        self.get_weights()
        self.get_biases()
    
    # Método para criar uma rede com pesos aleatórios
    # ex: [2, 4, 2] 3 camadas, uma com 2, a outra com 4 e a ultima com 2 neuronios
    def create_random(self, number_inputs, layers):
        self.number_inputs = number_inputs
        layer_inputs = self.number_inputs
        for number_perceptrons in layers:
            self.layers.append(Layer(self.activation_function, layer_inputs, number_perceptrons))
            layer_inputs = number_perceptrons
        self.get_weights()
        self.get_biases()
    
    def forward(self,inputs):
        outputs = []#[inputs]
        current_input = inputs
        for layer in self.layers:
            current_input = layer.forward(current_input)
            outputs.append(current_input)
        return outputs
        #return current_input
    
    def get_error(self, inputs, target_outputs):
        outputs = self.forward(inputs)

        # Erro Absoluto (SAE)
        # error = outputs - target_outputs

        # Erro Quadrático Médio (MSE)
        error = (1/2)*(outputs - target_outputs)^2
        
        return error
    
    def get_error_derivative(self, inputs, target_outputs):
        outputs = self.forward(inputs)
        self.error = (1/2)*(outputs - target_outputs)**2

        error_derivative = (outputs - target_outputs)

        return error_derivative
    
    
    def get_weights(self):
        self.weights = []
        for layer in self.layers:
            self.weights.append([layer.get_weights()])
        self.weights = np.array(self.weights, dtype=object)

    def put_weights(self):
        for index, layer in enumerate(self.layers):
            layer.put_weights(self.weights[index])
    
    def get_biases(self):
        self.biases = []
        for layer in self.layers:
            self.biases.append([layer.get_biases()])
        #self.weights = np.array(self.weights, dtype=object)
    
    def put_biases(self):
        for index, layer in enumerate(self.layers):
            layer.put_biases(self.biases[index])
    
    
    # Método para salvar uma rede em um arquivo .json
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

    def show(self):
        for index, layer in enumerate(self.layers):
            print(f"Layer {index+1}: ")
            layer.show()

    

# Classe Layer
class Layer:
    def __init__(self, activation_function, number_inputs, perceptrons):
        self.perceptrons = []
        self.outputs = []
        if isinstance(perceptrons, int):
            for _ in range(perceptrons):
                self.perceptrons.append(Perceptron(activation_function, number_inputs))
        else:
            for perceptron in perceptrons:
                self.perceptrons.append(Perceptron(activation_function, number_inputs, perceptron["weights"], perceptron["bias"]))
    
    def forward(self, inputs):
        outputs = np.array([perceptron.predict(inputs) for perceptron in self.perceptrons])
        self.outputs = outputs
        return outputs


    def show(self):
        for index, perceptron in enumerate(self.perceptrons):
            print(f"    Perceptron {index+1}: ", end='')
            perceptron.show()

    def get_weights(self):
        layer_weights = np.array([])
        for perceptron in self.perceptrons:
            for weight in perceptron.weights:
                layer_weights = np.append(layer_weights, weight)
        return layer_weights
    
    def put_weights(self, weights):
        i = 0
        for perceptron in self.perceptrons:
            for j in range(len(perceptron.weights)):
                perceptron.weights[j] = weights[0][i]
                i = i+1

    def get_biases(self):
        layer_biases = []
        for perceptron in self.perceptrons:
            layer_biases.append(perceptron.bias)
        return layer_biases
    
    def put_biases(self, biases):
        i = 0
        for perceptron in self.perceptrons:
                perceptron.bias = biases[0][i]
                i = i+1


# Classe perceptron
class Perceptron:
    def __init__(self,activation_function, number_inputs, weights=None, bias=None):
        self.number_inputs = number_inputs
        self.activation_function = activation_function
        if(weights is None):
            self.weights = np.random.rand(number_inputs)
        else:
            self.weights = weights
        if(bias is None):
            self.bias = np.random.rand()
        else:
            self.bias = bias

    # Calcula a saida do percepton a partir das entradas e dos seus proprios pesos
    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        output = self.activation_function(weighted_sum)
        return output
    
    def show(self):
        print(f"        weights: {self.weights} - bias: {self.bias}")

# Função de ativação, nesse caso to usando uma step funcion
def step_function(x):
    #return x
    return 1 if x >= 0 else 0

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # f = sigmoid_function(x)
    return x * (1 - x)


if __name__ == "__main__":
     # Cria uma rede, precisa passar a função de ativação como parametro
    net = Network()
    # current = Layer.forward()
    # erro = Network.get_error()

    # Network.backpropagation(, learning_rate=0.1)

    # Abre uma rede a partir de um arquivo .json
    # net.load('./networks/network1.json') 
    
    # Cria uma rede com os pesos e bias aleatórios
    # o primeiro valor é quantidade de inputs
    # o segundo valor é uma lista com a quantidade de neuronios por camada
    # ex: [2, 4, 2] 3 camadas, uma com 2, a outra com 4 e a ultima com 2 neuronios
    net.create_random(2, [2, 1]) 
    #net.load("./networks/network2.json")
    
    # Mostra os neuronios de uma camada
    #for layer in net.layers:
    #    layer.show()

    # net.layers[0].perceptrons[0].show()
    # print("camada 1")
    # print(f"output perceptron1: {net.layers[0].perceptrons[0].predict([1, 2])}")
    # print(f"output perceptron2: {net.layers[0].perceptrons[1].predict([1, 2])}")
    # print(f"output layer: {net.layers[0].forward([1, 2])}")
    # output = net.layers[0].forward([1, 2])

    # print("Camada 2")
    # print(f"output perceptron1: {net.layers[1].perceptrons[0].predict(output)}")
    # print(f"output perceptron2: {net.layers[1].perceptrons[1].predict(output)}")
    # print(f"output layer: {net.layers[1].forward(output)}")

    # print(f"output da rede: {net.forward([1, 2])}")

    # print(f"error da rede: {net.get_error([1, 2], [0, 0])}")

    #Salva a rede nun arquivo .json
    net.save("./networks/network2.json")