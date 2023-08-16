import numpy as np
import json

# Classe network
class Network:
    def __init__(self, activation_function):
        self.layers = []
        self.number_inputs = 0
        self.activation_function = activation_function

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
    
    # Método para criar uma rede com pesos aleatórios
    def create_random(self, number_inputs, layers):
        self.number_inputs = number_inputs
        layer_inputs = self.number_inputs
        for number_perceptrons in layers:
            self.layers.append(Layer(self.activation_function, layer_inputs, number_perceptrons))
            layer_inputs = number_perceptrons
    
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

# Classe Layer
class Layer:
    def __init__(self, activation_function, number_inputs, perceptrons):
        self.perceptrons = []
        if isinstance(perceptrons, int):
            for _ in range(perceptrons):
                self.perceptrons.append(Perceptron(activation_function, number_inputs))
        else:
            for perceptron in perceptrons:
                self.perceptrons.append(Perceptron(activation_function, number_inputs, perceptron["weights"], perceptron["bias"]))
    
    def show(self):
        for index, perceptron in enumerate(self.perceptrons):
            print(f"Perceptron {index+1}: ", end='')
            perceptron.show()


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
        print(f"weights: {self.weights} - bias: {self.bias}")

# Função de ativação, nesse caso to usando uma step funcion
def step_function(x):
    return 1 if x >= 0 else 0

if __name__ == "__main__":
     # Cria uma rede, precisa passar a função de ativação como parametro
    net = Network(activation_function=step_function)

    # Abre uma rede a partir de um arquivo .json
    # net.load('./networks/network1.json') 
    
    # Cria uma rede com os pesos e bias aleatórios
    # o primeiro valor é quantidade de inputs
    # o segundo valor é uma lista com a quantidade de neuronios por camada
    # ex: [2, 4, 2] 3 camadas, uma com 2, a outra com 4 e a ultima com 2 neuronios
    net.create_random(2, [1]) 
    
    # Mostra os neuronios de uma camada
    net.layers[0].show()

    #Salva a rede nun arquivo .json
    net.save("./networks/network2.json")

    
