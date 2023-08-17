import network
import training
import matplotlib.pyplot as plt

net = network.Network()
net.create_random(2, [2, 1])
#net.load("./networks/network2.json")
#net.save("./networks/network3.json")

#primeira tabela
tr = training.Training([[[0,0],[0]], 
						[[0,1],[0]], 
                        [[1,0],[1]], 
                        [[1,1],[1]]], net, 0.1)

#errors, weight_list = tr.training_with_threshold(0.0001)
errors, weight_list = tr.training_with_limit(1000, False)


weight0 = []
weight1 = []
weight2 = []
weight3 = []
weight4 = []
weight5 = []

for all_weights in weight_list:
    weight0.append(all_weights[0])
    weight1.append(all_weights[1])
    weight2.append(all_weights[2])
    weight3.append(all_weights[3])
    weight4.append(all_weights[4])
    weight5.append(all_weights[5])
    
fig, ax1 = plt.subplots()

ax1.plot(range(len(weight0)), weight0, label = 'peso0')
ax1.plot(range(len(weight1)), weight1, label = 'peso1')
ax1.plot(range(len(weight2)), weight2, label = 'peso2')
ax1.plot(range(len(weight3)), weight3, label = 'peso3')
ax1.plot(range(len(weight4)), weight4, label = 'peso4')
ax1.plot(range(len(weight5)), weight5, label = 'peso5')
ax1.set_ylabel('Pesos', color='black')
ax1.set_xlabel('Épocas')

ax2 = ax1.twinx()

plt.title('Evolução dos pesos e do erro')

ax2.plot(range(len(errors)), errors, label = 'Erro', linestyle='dashed')
ax2.set_ylabel('Erro', color='black')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

plt.show()
