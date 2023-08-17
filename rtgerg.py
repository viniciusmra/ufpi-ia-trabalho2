import matplotlib.pyplot as plt
import numpy as np

# Dados para o eixo x
x = np.linspace(0, 10, 100)

# Dados para a primeira curva (usando o mesmo eixo y)
y1 = np.sin(x)

# Dados para a segunda curva (usando um segundo eixo y)
y2 = np.cos(x) * 10  # Multiplicado por 10 para fins de escala

# Criar a figura e o primeiro eixo y
fig, ax1 = plt.subplots()

# Plot da primeira curva no primeiro eixo y
ax1.plot(x, y1, label='Curva 1 - Seno', color='blue', linestyle='dashed')
ax1.set_xlabel('Eixo X')
ax1.set_ylabel('Eixo Y (Curva 1)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Criar o segundo eixo y
ax2 = ax1.twinx()

# Plot da segunda curva no segundo eixo y
ax2.plot(x, y2, label='Curva 2 - Cosseno', color='red')
ax2.set_ylabel('Eixo Y (Curva 2)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Título do gráfico
plt.title('Gráfico com Dois Eixos Y')

# Adicionar legendas para as curvas
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

# Exibir o gráfico
plt.show()